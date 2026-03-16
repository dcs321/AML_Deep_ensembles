# Data loading and preprocessing
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


from PIL import Image, UnidentifiedImageError

#CLASSIFICATION

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


def load_notmnist(batch_size=100, root='./datasets/classification/notMNIST_small'):
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=root, transform=transformation)

    # skip invalid/corrupted images
    valid_samples = []
    removed = 0
    for path, label in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_samples.append((path, label))
        except (UnidentifiedImageError, OSError, ValueError):
            removed += 1

    dataset.samples = valid_samples
    dataset.imgs = valid_samples

    print(f"Removed {removed} corrupted images from NotMNIST")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


#REGRESSION

def load_boston_housing(num_of_train_test_splits, train_ratio_in_split, batch_size=100):
    # TODO
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]

    n_samples = len(X)
    rng = np.random.RandomState(42)
    n_train   = int(np.floor(train_ratio_in_split * n_samples))
    train_loaders = [] 
    test_loaders = [] 
    output_means = [] 
    output_stds = []

    for _ in range(num_of_train_test_splits):
        perm      = rng.permutation(n_samples)
        train_idx = perm[:n_train]
        test_idx  = perm[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train_n = ((X_train - X_mean) / (X_std + 1e-8)).astype(np.float32)
        X_test_n  = ((X_test  - X_mean) / (X_std + 1e-8)).astype(np.float32)

        y_mean, y_std = float(y_train.mean()), float(y_train.std())
        y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
        y_test_n  = ((y_test  - y_mean) / y_std).astype(np.float32)

        train_loaders.append(DataLoader(TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train_n)), batch_size=batch_size, shuffle=True))
        test_loaders.append( DataLoader(TensorDataset(torch.from_numpy(X_test_n),  torch.from_numpy(y_test_n)),  batch_size=batch_size, shuffle=False))
        output_means.append(y_mean)
        output_stds.append(y_std)

    return train_loaders, test_loaders, output_means, output_stds
