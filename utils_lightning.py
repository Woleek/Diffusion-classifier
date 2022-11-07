import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
 
def decode_one_hot(y: NDArray) -> NDArray:
    """Decodes labels from one-hot encoding to integers.

    Args:
        y (NDArray): array of labels encoded with one-hot

    Returns:
        NDArray: array of labels as integers
    """
    y_labels = np.array([np.argmax(i) for i in y])
    return y_labels

def load_train_data(batch=1) -> Tuple[TensorDataset, DataLoader]:
    """Loads training dataset with labels. Transforms numpy arrays to torch tensors.
    Shuffles input data.

    Args:
        batch (int, optional): Size of batch for setting dataloader. Defaults to 1.

    Returns:
        Tuple[TensorDataset, DataLoader]: dataset and dataloader for training dataset
    """
    data = np.load("data/X_train.npy") # (49000, 300, 2) = N, L, P(x,y)
    labels = np.load("data/y_train.npy") # (49000, 5) = N, labels (OneHot)
    labels = decode_one_hot(labels) # (49000, 5) -> (49000, 1)
    
    tensor_x = torch.FloatTensor(data)
    tensor_y = torch.LongTensor(labels)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch, 
                            num_workers=4,
                            shuffle=True)
    return dataset, dataloader
    
def load_val_data(batch=1) -> Tuple[TensorDataset, DataLoader]:
    """Loads validation dataset with labels. Transforms numpy arrays to torch tensors.
    Does not shuffle input data.

    Args:
        batch (int, optional): Size of batch for setting dataloader. Defaults to 1.

    Returns:
        Tuple[TensorDataset, DataLoader]: dataset and dataloader for validation dataset
    """
    data = np.load("data/X_val.npy")
    labels = np.load("data/y_val.npy")
    labels = decode_one_hot(labels)
    
    tensor_x = torch.FloatTensor(data)
    tensor_y = torch.LongTensor(labels)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch,
                            num_workers=4,
                            shuffle=False)
    return dataset, dataloader

def load_test_data(batch=1) -> Tuple[TensorDataset, DataLoader]:
    """Loads validation dataset without labels. Transforms numpy array to torch tensor.
    Does not shuffle input data.
    
    Args:
        batch (int, optional): Size of batch for setting dataloader. Defaults to 1.

    Returns:
        Tuple[TensorDataset, DataLoader]: dataset and dataloader for test dataset
    """
    data = np.load("data/X_test.npy")
    
    tensor_x = torch.FloatTensor(data)
    
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch,
                            num_workers=4,
                            shuffle=False)
    return dataset, dataloader