import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict

import utils_lightning

# Fully connected neural network with one hidden layer
class LitModel(pl.LightningModule):
    """Klasa implementująca model klasyfikatora dyfuzji. 
    """
    def __init__(self, num_layers:int, hidden_size:int, input_size:int, num_classes:int, batches: Dict[str, int]) -> None:
        """Inicjalizacja atrybutów klasy.

        Args:
            num_layers (int): liczba połączynych sieci neuronowych
            hidden_size (int): liczba ukrytych warstw (ustawiających wagi połączeń)
            input_size (int): liczba wejść
            num_classes (int): liczba wyjść
            batches (Dict[str, int]): wielkość partii danych dla train, val i test
        """
        super(LitModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.batches = batches
        # x needs to be: (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.train_ac = torchmetrics.Accuracy()
        self.val_ac = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        
        hidden_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # x: (N, 300, 2), hidden_init: (num_layers, N, hidden_size)
        
        out, _ = self.model(x, hidden_init)
        # out: tensor of shape (batch_size, seqence_length, hidden_size)
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)
        out = self.fc(out)
        # out: (batch_size, 5)
        return out

    def train_dataloader(self):
        train_dataset, train_dataloader = utils_lightning.load_train_data(self.batches['train_bs'])
        return train_dataloader

    def val_dataloader(self):
        val_dataset, val_dataloader = utils_lightning.load_val_data(self.batches['val_bs'])
        return val_dataloader
    
    def training_step(self, batch, batch_idx):
        stack_of_points, labels = batch
        
        # forward pass
        outputs = self(stack_of_points)
        self.train_ac(outputs, labels)
        loss = F.cross_entropy(outputs, labels)
        
        self.log({'train_loss': loss, 'train_acc': self.train_ac*100}, prog_bar=True)
        return {"loss": loss, "acc": self.train_ac}
    
    def validation_step(self, batch, batch_idx):
        stack_of_points, labels = batch
        
        # forward pass
        outputs = self(stack_of_points)
        loss = F.cross_entropy(outputs, labels)
        self.val_ac(outputs, labels)
        
        self.log_dict({'val_loss': loss, 'val_acc': self.val_ac}, on_epoch=True)
        return {'loss': loss, 'acc': self.val_ac}

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=True, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}