import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import panda as pd
import joblib
import os


# --Config--
SEED = 33
DATA_PATH = ""
SAVE_DIR = ""
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES=[
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

TARGET = 'Machine failure'

MLP_HIDDEN = [128,64]
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_BATCH = 64
MLP_PATIENCE = 10


# --Device detection--
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else :
        return torch.device("cpu")


# --Data load and cleab--
def load_data(PATH=DATA_PATH):
    pass

def clean_data(df):
    pass

def prepare_data(df):
    pass

def split_data(df):
    pass


# --MLP PyTorch--

# --RF Sickit Learn--

# --Compare MLP vs RF--




# --Main function call and menu--
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device : {device}")


if __name__ == '__main__':
    main()