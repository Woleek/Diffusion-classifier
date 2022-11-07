import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchmetrics
from typing import Dict

import utils_lightning

# Fully connected neural network with one hidden layer
class LitModel(pl.LightningModule):
    """Class implementation of diffusion classifier model. It takes number of sequences of points as input and returns 1 of 5 classes based on that sequence.
    """
    def __init__(self, num_layers:int, hidden_size:int, input_size:int, num_classes:int, batches: Dict[str, int]) -> None:
        """Class atributes initialization.

        Args:
            num_layers (int): number of stacked neural networks
            hidden_size (int): number of hidden layers (calculating weights)
            input_size (int): number of inputs (point coordinates)
            num_classes (int): number of outputs (possible classes)
            batches (Dict[str, int]): size of batches for train, validation and test dataloaders
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward function, that gets output (predicted class) from last neuron of recurrent naural network.

        Args:
            x (torch.Tensor): batch of input sequences

        Returns:
            torch.Tensor: predicted classes for sequences in batch
        """
        hidden_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # x: (N, 300, 2), hidden_init: (num_layers, N, hidden_size)
        
        out, _ = self.model(x, hidden_init)
        # out: tensor of shape (batch_size, seqence_length, hidden_size)
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)
        out = self.fc(out)
        # out: (batch_size, 5)
        return out

    def train_dataloader(self) -> DataLoader:
        """Loads training dataset.

        Returns:
            DataLoader: dataloader for training dataset
        """
        train_dataset, train_dataloader = utils_lightning.load_train_data(self.batches['train_bs'])
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        """Loads validation dataset.

        Returns:
            DataLoader: dataloader for validatipn dataset
        """
        val_dataset, val_dataloader = utils_lightning.load_val_data(self.batches['val_bs'])
        return val_dataloader
    
    def training_step(self, batch:list, batch_idx:int) -> Dict[str, float]:
        """Training step function of neural network. Predicts labels from batch of sequences, then calculates accurancy and loss based on actual labels.

        Args:
            batch (list): batch of inputs
            batch_idx (int): index of current batch

        Returns:
            Dict[str, float]: current loss and accuracy of training
        """
        stack_of_points, labels = batch
        
        # forward pass
        outputs = self(stack_of_points)
        self.train_ac(outputs, labels)
        loss = F.cross_entropy(outputs, labels)
        
        self.log_dict({'train_loss': loss, 'train_acc': self.train_ac}, prog_bar=True)
        return {"loss": loss, "acc": self.train_ac}
    
    def validation_step(self, batch:list, batch_idx:int) -> Dict[str, float]:
        """Validation step function of neural netrowk. Calculates model development performence, logs validation loss and accuracy at the end of every epoch.

        Args:
            batch (list): batch of inputs
            batch_idx (int): index of current batch

        Returns:
            Dict[str, float]: current loss and accuracy of validation
        """
        stack_of_points, labels = batch
        
        # forward pass
        outputs = self(stack_of_points)
        loss = F.cross_entropy(outputs, labels)
        self.val_ac(outputs, labels)
        
        self.log_dict({'val_loss': loss, 'val_acc': self.val_ac}, on_epoch=True)
        return {'loss': loss, 'acc': self.val_ac}

    
    def configure_optimizers(self) -> Dict[str, any]:
        """Function that sets optimizer and learning rate scheduler.

        Returns:
            Dict[str, any]: optimizer and lr_scheduler set for udage
        """
        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=True, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}