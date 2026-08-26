#from fastai.vision.all import *

import pandas as pd
import numpy as np
from torch import tensor
import torch

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

def dldat(file='train.csv', path=path):
    fullpath = path + '/' + file
    df = pd.read_csv(fullpath)
    return df

def dataclean(df):
    #identify missing var
    df.isna().sum()
    #identify the most common value for each columns
    modes = df.mode().iloc[0]
    #replace missing value by modes
    df.fillna(modes, inplace=True)
    return df


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

    n_coeff = t_indep.shape[1]
    coeffs = torch.rand(n_coeff)-0.5



    
    #plot sigmoid, base setup for future test
    x = sp.symbols('x')
    f1 = 1/(1+sp.exp(-x))
    #sp.plot(f1, xlim=(-5,5), show=True)

if __name__ == '__main__':
    main()