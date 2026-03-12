# Data loading and preprocessing
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=100, download=True, with_validation_set=False):
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./datasets/classification', train=True, transform=transformation, download=download)
    if with_validation_set:
        num_train_samples = int(0.8 * len(train_dataset))
        num_val_samples = len(train_dataset) - num_train_samples
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train_samples, num_val_samples])
   
    test_dataset = datasets.MNIST(root='./datasets/classification', train=False, transform=transformation, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if with_validation_set:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if with_validation_set:
        return train_loader, val_loader, test_loader
    
    return train_loader, None, test_loader