import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
 
def decode_one_hot(y):
    y_labels = np.array([np.argmax(i) for i in y])
    return y_labels

def load_train_data(batch=1):
    data = np.load("X_train.npy") # (49000, 300, 2) = N, L, P(x,y)
    labels = np.load("y_train.npy") # (49000, 5) = N, labels (OneHot)
    labels = decode_one_hot(labels) # (49000, 5) -> (49000, 1)
    
    tensor_x = torch.FloatTensor(data)
    tensor_y = torch.LongTensor(labels)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch, 
                            num_workers=4,
                            shuffle=True)
    return dataset, dataloader
    
def load_val_data(batch=1):
    data = np.load("X_val.npy")
    labels = np.load("y_val.npy")
    labels = decode_one_hot(labels)
    
    tensor_x = torch.FloatTensor(data)
    tensor_y = torch.LongTensor(labels)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch,
                            num_workers=4,
                            shuffle=False)
    return dataset, dataloader

def load_test_data(batch=1):
    data = np.load("X_test.npy")
    
    tensor_x = torch.FloatTensor(data)
    
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch,
                            num_workers=4,
                            shuffle=False)
    return dataset, dataloader