#from fastai.vision.all import *

import pandas as pd
import numpy as np
from torch import tensor
import torch

from sklearn.model_selection import train_test_split

import time
import sympy as sp


#global variables
path = 'csv'

#csv load data from path
def dldat(file='train.csv', path=path):
    fullpath = path + '/' + file
    df = pd.read_csv(fullpath)
    return df

#csv clean data
def dataclean(df):
    #identify missing var
    df.isna().sum()
    #identify the most common value for each columns
    modes = df.mode().iloc[0]
    #replace missing value by modes
    df.fillna(modes, inplace=True)
    return df


#take 2 tensors and split them between train, valid and test sets
def splitdata(t_indep, t_dep, train_frac=0.7, valid_frac=0.15, test_frac=0.15, seed=33):
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-6, "total frac must alway be 1"

    idx_train, idx_temp = train_test_split(
        np.arange(len(t_indep)),
        train_size=train_frac,
        random_state=seed
    )

    relative_valid = valid_frac / (valid_frac + test_frac)
    idx_valid, idx_test = train_test_split(
        idx_temp,
        test_size=relative_valid,
        random_state=seed
    )

    return{
        'train':(t_indep[idx_train],t_dep[idx_train]),
        'valid':(t_indep[idx_valid],t_dep[idx_valid]),
        'test': (t_indep[idx_test],t_dep[idx_test])
    }



def calc_preds(coeffs, indeps):
    return (indeps*coeffs).sum(axis=1, keepdim=True)

def calc_preds_sigmoid(coeffs, indeps):
    return torch.sigmoid((indeps*coeffs).sum(axis=1, keepdim=True))

def calc_loss(coeffs, indeps, deps):
    return torch.abs(calc_preds(coeffs, indeps)-deps).mean()


def calc_loss_bce(coeffs, indeps, deps, pos_weight=28.0):
    preds = calc_preds_sigmoid(coeffs, indeps)
    eps = 1e-7
    preds = preds.clamp(eps, 1-eps)
    loss = -(pos_weight*deps*torch.log(preds) + (1-deps)*torch.log(1-preds))
    return loss.mean()


def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr),
    coeffs.grad.zero_()

def one_epoch(epoch, coeffs, lr, trn_indep, trn_dep):
    loss = calc_loss_bce(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"epoch : {epoch}, loss : {loss:.3f}")


def init_coeffs(n_coeff): 
    return (torch.rand(n_coeff)-0.5).requires_grad_()


def train_model(epochs, lr, n_coeff, trn_indep, trn_dep):
    torch.manual_seed(442)
    coeffs = init_coeffs(n_coeff)
    for i in range(epochs): 
        one_epoch(i, coeffs, lr=lr, trn_indep=trn_indep, trn_dep=trn_dep)
    return coeffs


def main():
    #get csv and basic cleanup
    df = dldat('ai4i2020.csv', 'dataset')
    df = dataclean(df)

    #check data summary
    #print(df.describe(include=(np.number)))
    #check non numerical data and create type for revelant one
    #print(df.describe(include=(str)))
    df = pd.get_dummies(df, columns=["Type"])

    #select variables we keep for prediction
    selected_col = ['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_H',
       'Type_L', 'Type_M']


    #create tensors

    #create dependent variable
    t_dep = tensor(df['Machine failure'])
    #convert to 2D tensor
    t_dep = t_dep.unsqueeze(1)
    #print(t_dep.shape)

    #create independent variables
    t_indep = tensor(df[selected_col].to_numpy(dtype=np.float32), dtype=torch.float32)
    #print(t_indep.shape)


    #setup linear model
    torch.manual_seed(33)

    #create random coeffs
    n_coeff = t_indep.shape[1]
    coeffs = torch.rand(n_coeff)-0.5

    #dirty matrix mult to uniformize values
    vals,indices = t_indep.max(dim=0)
    t_indep = t_indep / vals


    splits = splitdata(t_indep, t_dep, train_frac=0.7, valid_frac=0.15, test_frac=0.15)

    t_indep_train, t_dep_train = splits['train']
    t_indep_valid, t_dep_valid = splits['valid']
    t_indep_test, t_dep_test = splits['test']

    coeffs = train_model(epochs=100, lr=0.01, n_coeff=n_coeff, trn_indep=t_indep_train, trn_dep=t_dep_train)
    print(coeffs)


    preds = calc_preds_sigmoid(coeffs, t_indep_valid)
    results = t_dep_valid.bool()==(preds>0.5)
    print(results[:16])
    print(results.float().mean())


    #preds = calc_preds_sigmoid(coeffs, t_indep_valid)
    preds_bin = preds > 0.5

    tp = ((preds_bin==True) & (t_dep_valid.bool()==True)).sum().item()
    fn = ((preds_bin==False) & (t_dep_valid.bool()==True)).sum().item()
    fp = ((preds_bin==True) & (t_dep_valid.bool()==False)).sum().item()

    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    print(f"TP={tp}, FN={fn}, FP={fp}, recall={recall:.3f}")
    
    #plot sigmoid, base setup for future test
    x = sp.symbols('x')
    f1 = 1/(1+sp.exp(-x))
    #sp.plot(f1, xlim=(-5,5), show=True)

if __name__ == '__main__':
    main()