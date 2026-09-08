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

#get CSV data set
#clean data set
#split data set in train, cross, test
# 
#define archi MLP pytorch
#define train ad save
#
#define archi RF sickitlearn
#define train and save
#
#menu train, compare