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


# --------Device detection--------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else :
        return torch.device("cpu")


# --------Data load and cleab--------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"loaded : {len(df)} rows, {len(df.columns)} cols")

def clean_data(df):
    #search NaN numerical and replace by median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any() :
            df[col].fillna(df[col].median, inplace=True)

    #search for NaN string and repalce by most frequent
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    #create boolean value for exotic type ie. ["a","b","c"] 
    df = pd.get_dummies(df, columns=["Type"])
    print(f"cleaned : {df.shape[1]} cols")
    return df
    

def prepare_data(df):
    features_cols = FEATURES + [c for c in df.columns if c.startswith("Type_")]
    X = df[features_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.int64)

    #normalize features for MLP
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    #Z score 0 center feature
    X_norm = (X - mean) / std

    print(f"features : {X_norm.shape[1]} | samples : {len(X_norm)}")
    #count OK and FAIL occurence
    print(f"class balance : OK={np.bincount(y)[0]} | FAIL={np.bincount(y)[1]} "
          f"(ratio 1:{np.bincount(y)[0]/max(np.bincount(y)[1],1):.0f}:1)")
    
    return X_norm, y, features_cols

    
def split_data(X, y, train_frac=0.7, val_frac=0.15, seed=SEED):
    #we keep a split at 70 training, 15 cross, 15 test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_frac, random_state=seed, stratify=y
    )

    rel_val = val_frac / (val_frac + 0.15)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_val, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------MLP PyTorch--------

"""
    Simple MLP feedforward Class
    Input : 8 features (5 numeric + 3 types)
    Hidden : configurable layers, all ReLU with dropout
    Output : 1 neuron (logit + sigmoid)
"""
class TabularMLP(nn.Module):

    def __init__(self, input_dim, hidden_layers=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_layers :
            layers += [
                nn.Linear(prev, h),
                nn. Relu(),
                nn.Dropout(dropout)
                ]
            prev = h
        layers.append(nn.Linear(prev,1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #squeeze to delete last dimension for y target on 1D prediction
        return self.net(x).squeeze(-1)


def mlp_train_one_epoch(model, loader, criterion, optimizer, device):
    #put model in train mode
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader :
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        #forward pass
        logits = model(X_batch)
        #BCE with logistics loss
        loss = criterion(logits, y_batch)

        #reinit grad, we don't want grad accumulation here
        optimizer.zero_grad()
        #compute dl/dw
        loss.backward()
        #w = w - lr * grad
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss/total , correct/total


@torch.no.grad()
def mlp_evaluate(model, loader, criterion, device):
    pass

def mlp_train(X_train, y_train, X_val, y_val, device):
    pass

# --------RF Sickit Learn--------

# --------Compare MLP vs RF--------




# --------Main function call and menu--------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device : {device}")

    while True :
        print("\n" + "─"*40)
        print("  MACHINE FAILURE PREDICTION")
        print("─"*40)
        print("1: train MLP (PyTorch)")
        print("2: train Random Forest (sklearn)")
        print("3: evaluate MLP on test set")
        print("4: evaluate RF on test set")
        print("5: compare MLP vs RF")
        print("0: quit")

        choice = int(input("choose : "))

        if choice == 1:
            pass

        elif choice == 2:
            pass

        elif choice == 3:
            pass

        elif choice == 4:
            pass

        elif choice == 5:
            pass

        else:
            break


#dunder call encapsulation by security
if __name__ == '__main__':
    main()