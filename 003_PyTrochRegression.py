import pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# --- Data loading ---
path = 'dataset'
file = "ai4i2020.csv"
fullpath = path + '/' + file

df = pd.read_csv(fullpath)  


# --- Data Preparation ---

#clean data
df.isna().sum()
#identify the most common value for each columns
modes = df.mode().iloc[0]
#replace missing value by modes
df.fillna(modes, inplace=True)


#we choose to  drop columns we don't want to keep
X = df.drop(columns=["Machine failure", 
                     "TWF",
                     "HDF",
                     "PWF",
                     "OSF",
                     "RNF",
                     "Product ID"]).copy()

#create new dummies for Type as is is non numerical, also  put drop_first at true to avoid dummy variable trap
dummies = pd.get_dummies(X["Type"], prefix="type", drop_first=True, dtype=int)
X = pd.concat([X.drop(columns=["Type"]), dummies], axis=1)

#stock our desired prediction value
y = df["Machine failure"].values


# --- Data split bewteen training and test ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#fit our data for Xtr to define our scaler
scaler = StandardScaler().fit(Xtr)

#create our set for training and test, we scale Xte only by the scaler get from our training values
Xtr_t = torch.tensor(scaler.transform(Xtr), dtype=torch.float32)
Xte_t = torch.tensor(scaler.transform(Xte), dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(1)


# --- Model creation ---

#32 output for our Linear layer was choose for test purpose, we need to investigate on it
model = nn.Sequential(nn.Linear(scaler.n_features_in_, 32), 
                      nn.ReLU(),
                      nn.Linear(32, 1))

#get number of positive and negative y values in our ytr set
n_pos, n_neg = ytr.sum(), len(ytr) - ytr.sum()

#use a pos_weight to make our failure as important as our non failure as there are not much failure in our df
#BCEWithLogitsLoss = sigmoid + BCE
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))

#define our optimizer, Adam as usual and our learning rate
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


# --- Training Loop ---
epochs = 500

for epoch in range(epochs):
    #train our model on one epoch
    model.train()
    #clean previous gradient descent
    opt.zero_grad()
    #compute our loss
    loss = criterion(model(Xtr_t), ytr_t)
    #calculate our gradient for each param
    loss.backward()
    #call Adam to move our weight
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")


# --- Evaluation ---

#put our model in eval mode
model.eval()
with torch.no_grad():
    #proba in [0,1]
    probs = torch.sigmoid(model(Xte_t))
#put probs in a 1D vector
probs = probs.squeeze()
#prepare for sklearn
probs = probs.numpy()

#define preds at a 0.5 probability split
preds = (probs >= 0.5).astype(int)

#show preds report
print(classification_report(yte, preds))
#check probs prediction precision
print("ROC-AUC :", roc_auc_score(yte, probs))

"""
def main():
    pass

if __name__ == '__main__':
    main()
"""