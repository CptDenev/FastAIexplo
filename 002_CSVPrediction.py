#from fastai.vision.all import *

import pandas as pd
import numpy as np
from torch import tensor
import torch

from sklearn.model_selection import train_test_split

import time
import sympy as sp

#import csv as pd
#analysze and clean tabular dataset
#uniformize tabular

#split data set in 70/15/15 for train/cross/test

#create tensor for pytorch x -> data, y -> fail or not

#create nn architecture x entry, y exit, see for hidden layers

#create training loop with cross validation
#define learning rate

#train the model on e epochs
#test it

#save it 
#invoke it and use it

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
        np.arrange(len(t_indep)),
        train_size=train_frac,
        random_state=seed
    )

    relative_valid = train_frac - (valid_frac + test_frac)
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
    return (indeps*coeffs).sum(axis=1)



def calc_loss(coeffs, indeps, deps):
    return torch.abs(calc_preds(coeffs, indeps)-deps).mean()


def main():
    #get csv and basic cleanup
    df = dldat('ai4i2020.csv', 'dataset')
    df = dataclean(df)

    #check data summary
    print(df.describe(include=(np.number)))
    #check non numerical data and create type for revelant one
    print(df.describe(include=(str)))
    df = pd.get_dummies(df, columns=["Type"])

    #select variables we keep for prediction
    selected_col = ['UDI', 'Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
       'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type_H',
       'Type_L', 'Type_M']


    #create tensors

    #create dependent variable
    t_dep = tensor(df['Machine failure'])
    #convert to 2D tensor
    t_dep = t_dep.unsqueeze(1)
    print(t_dep.shape)

    #create independent variables
    t_indep = tensor(df[selected_col].to_numpy(dtype=np.float32), dtype=torch.float32)
    print(t_indep.shape)


    #setup linear model
    torch.manual_seed(33)

    #create random coeffs
    n_coeff = t_indep.shape[1]
    coeffs = torch.rand(n_coeff)-0.5

    #dirty matrix mult to uniformize values
    vals,indices = t_indep.max(dim=0)
    t_indep = t_indep / vals

    
    coeffs.requires_grad_()
    loss = calc_loss(coeffs, t_indep, t_dep)
    loss.backward()

    with torch.no_grad():
        coeffs.sub_(coeffs.grad * 0.1)
        coeffs.grad.zero_()
        print(calc_loss(coeffs, t_indep, t_dep))


    
    #plot sigmoid, base setup for future test
    x = sp.symbols('x')
    f1 = 1/(1+sp.exp(-x))
    #sp.plot(f1, xlim=(-5,5), show=True)

if __name__ == '__main__':
    main()