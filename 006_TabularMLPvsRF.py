import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import joblib
import os
import math


# --Config--
SEED = 33
DATA_PATH = "./dataset/ai4i2020.csv"
SAVE_DIR = "./checkpoints"
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
    return df

def clean_data(df):
    #search NaN numerical and replace by median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any() :
            df[col].fillna(df[col].median(), inplace=True)

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
                nn.ReLU(),
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


@torch.no_grad()
def mlp_evaluate(model, loader, criterion, device):
    #put model in eval mode
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        preds = (torch.sigmoid(logits)>0.5).float()
        #move tensors to CPU, convert to numpy and add only element to tab
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    #compute matching preds with matching labels and mean it
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss/len(loader.dataset), acc, all_preds, all_labels

def mlp_train(X_train, y_train, X_val, y_val, device):
    torch.manual_seed(SEED)

    #--tensors and data loader--
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH, shuffle=False)

    #--model--
    input_dim = X_train.shape[1]
    model = TabularMLP(input_dim, MLP_HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP parameters : {n_params}")

    #compute pos weight due to huge class imbalance in data
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.Tensor([math.sqrt(neg_count / pos_count)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos weight :{pos_weight.item():.1f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    #wait 3 non progression for loss and then apply a 0.5 factor to it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    #--training loop--
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc":[], "val_acc":[]}

    print("\n" + "="*60)
    print(f"{'epoch':<6}{'train loss':<12}{'train acc':<12}{'val loss':<12}{'val acc':<12}")
    print("="*60)

    for epoch in range(1, MLP_EPOCHS +1):
        #train and evaluate for one epoch
        tr_loss, tr_acc = mlp_train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = mlp_evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"{epoch:<6}{tr_loss:<12.4f}{tr_acc:<12.4f}{val_loss:<12.4f}{val_acc:<12.4f}")

        #check if validation loss is better than previous one and save checkpoint as best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mlp_machfail_best.pth"))

        #else increase patience counter until we reach max value
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                print(f"early stop at epoch: {epoch} patience={MLP_PATIENCE}")
                break

    #--save model--
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_layers": MLP_HIDDEN,
        "best_val_loss": best_val_loss,
        "epoch": epoch,
        "feature_names": FEATURES + [f"Type_{t}" for t in ["H","L","M"]],
    }, os.path.join(SAVE_DIR, "mlp_machfail_final.pth"))

    return model, history
    


# --------RF Sickit Learn--------

def rf_train(X_train, y_train, X_val, y_val):
    """
        Sickit RF
        First run with 4 estimators 
        then we keep the best for a final run
    """
    print("\n" + "="*60)
    print("Random Forest - progressive n-estimators")
    print("="*60)

    best_f1, best_n = 0, 100
    #test multiple n estimators
    for n_est in [50, 100, 200, 400]:
        rf = RandomForestClassifier(
             n_estimators=n_est,
             max_depth=None,
             min_samples_split=5,
             class_weight='balanced',
             random_state=SEED,
             n_jobs=1
        )

        rf.fit(X_train, y_train)

        val_pred = rf.predict(X_val)
        #we use f1 score as our y to predict are hugely asymetrics (way more running OK than running error)
        val_f1 = f1_score(y_val, val_pred)
        print(f"n_est={n_est:<5} | val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_n = val_f1, n_est

    #run with best estimator
    print(f"\n best n_estimator : {best_n}, val F1 : {best_f1:.4f}")
    rf_final = RandomForestClassifier(
        n_estimators=best_n,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=1
    )
    rf_final.fit(X_train, y_train)

    joblib.dump(rf_final, os.path.join(SAVE_DIR, "rf_machfail_best.joblib"))
    print(f"saved {SAVE_DIR}/rf_machfail_best.joblib")
    return rf_final


def rf_evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\nRF F1 : {f1_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["OK", "FAILURE"]))
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))

    imp = pd.Series(model.feature_importances_,
                    index=model.feature_names_in_).sort_values(ascending=False)
    print("\nfeature importance top 5 :")
    for name, val in imp.head(5).items():
        print(f"  {name:<30} {val:.4f}")
    return preds
    


# --------Compare MLP vs RF--------

def compare_models(X_test, y_test, device):
    print("\n" + "="*60)
    print(" COMPARISON : MLP VS RF on test set")
    print("="*60)


    #load anc compute preds for MLP
    ckpt = torch.load(os.path.join(SAVE_DIR, "mlp_machfail_final.pth"))
    mlp = TabularMLP(ckpt["input_dim"], ckpt["hidden_layers"]).to(device)
    mlp.load_state_dict(ckpt["model_state_dict"])
    mlp.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        mlp_probs = torch.sigmoid(mlp(X_test_t)).cpu().numpy()
    mlp_preds =(mlp_probs > 0.5).astype(int)

    #load and compute preds for RF
    rf = joblib.load(os.path.join(SAVE_DIR, "rf_machfail_best.joblib"))
    rf_preds = rf.predict(X_test)

    #convert our test in [0,1]
    y_test = y_test.astype(int)

    #display analysis F1 score, accuracy, precision, recall
    print(f"\n{'Metric':<20}{'MLP':<15}{'RF':<15}")
    print("-"*50)
    mlp_f1 = f1_score(y_test,mlp_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    print(f"{'F1 Score':<20}{mlp_f1:<15.4f}{rf_f1:<15.4f}")
    print(f"{'Accuracy':<20}{(mlp_preds == y_test).mean():15.4f}{(rf_preds == y_test).mean():15.4f}")

    mlp_report = classification_report(y_test, mlp_preds, target_names=["OK", "FAILURE"], output_dict=True)
    rf_report = classification_report(y_test, rf_preds, target_names=["OK", "FAILURE"], output_dict=True)

    print(f"\n{'':<20}{'MLP':<15}{'RF':<15}")
    print("="*50)
    for cls in ["OK", "FAILURE"]:
        print(f"{cls+' precision':<20}{mlp_report[cls]['precision']:<15.4f}{rf_report[cls]['precision']:<15.4f}")
        print(f"{cls+' recal':<20}{mlp_report[cls]['recall']:<15.4f}{rf_report[cls]['recall']:<15.4f}")
        print(f"{cls+' F1':<20}{mlp_report[cls]['f1-score']:<15.4f}{rf_report[cls]['f1-score']:<15.4f}")

    #display disagree
    disagree = (mlp_preds != rf_preds)
    print(f"\ndisagreement : {disagree.sum()} samples({disagree.mean()*100:.1f}%)")
    if disagree.sum() > 0:
        print(f" of which actual FAILURE : {y_test[disagree].sum()}")

    #display feature importance for RF
    imp = pd.Series(rf.feature_importances_,
                    index=rf.feature_names_in_).sort_values(ascending=False)

    print("RF feature importance :")
    for name, val in imp.items():
        bar = "O" * int(val*50)
        print(f" {name:<30} {val:.4f} {bar}")


# --------Main function call and menu--------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print(f"device : {device}")

    #load data from CSV and create test, val and test sets
    df = load_data()
    df = clean_data(df)
    X, y, feature_cols = prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

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
            mlp_train(X_train, y_train, X_val, y_val, device)

        elif choice == 2:
            X_train_df = pd.DataFrame(X_train, columns=feature_cols)
            X_val_df = pd.DataFrame(X_val, columns=feature_cols)
            rf_train(X_train_df, y_train, X_val_df, y_val)

        elif choice == 3:
            ckpt = torch.load(os.path.join(SAVE_DIR, "mlp_machfail_final.pth"),
                              map_location=device, weights_only=True)

            mlp = TabularMLP(ckpt["input_dim"], ckpt["hidden_layers"]).to(device)
            mlp.load_state_dict(ckpt["model_state_dict"])

            val_ds = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32))
            test_loader = DataLoader(val_ds, batch_size=MLP_BATCH, shuffle=False)

            pos_weight = torch.tensor([math.sqrt((len(y_test)-y_test.sum()) / max(y_test.sum(),1))]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            _, acc, preds, labels = mlp_evaluate(mlp, test_loader, criterion, device)

            print(f"MLP test accuracy: {acc:.4f}")
            print(classification_report(labels, preds, target_names=["OK", "FAILURE"]))
            print(confusion_matrix(labels, preds))

        elif choice == 4:
            rf = joblib.load(os.path.join(SAVE_DIR, "rf_machfail_best.joblib"))
            X_test_df = pd.DataFrame(X_test, columns=feature_cols)
            rf_evaluate(rf, X_test_df, y_test)

        elif choice == 5:
            compare_models(X_test, y_test, device)

        else:
            break


#dunder call encapsulation by security
if __name__ == '__main__':
    main()