import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

#Base setup and variables
SEED = 33
torch.manual_seed(SEED)
np.random.seed(SEED)

#check cuda or mps availability, else cpu
if(torch.cuda.is_available()):
    device = torch.device("cuda")
elif(torch.backends.mps.is_available()):
    device = torch.device("mps")
else:
    device = torch.device("cpu")


print(f"device on which we're running: {device}")

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
PATIENCE = 5
DATA_DIR = "./dataset/mnist"


#Load MNIST and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

print(f"train set : {len(train_dataset)} images")
print(f"test set : {len(test_dataset)} images")


#Split train in train + eval
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator = torch.Generator.manual_seed(SEED)
)

print(f"train set : {len(train_subset)} | validation set : {len(val_subset)} | test set {len(test_dataset)}")


#define data loaders
