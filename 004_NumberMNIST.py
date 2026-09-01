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

if(torch.cuda.is_available()):
    device = torch.device("cuda")
elif(torch.backends.mps.is_available()):
    device = torch.device("mps")
else:
    device = torch.device("cpu")


print(f"Device on which we're running: {device}")
print(torch.version.cuda)

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
PATIENCE = 5
DATA_DIR = "./dataset/mnist"