import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from UNetModel import UNet
from torchgeo.datasets import LoveDA


#---Config---
SEED = 33
DATA_PATH = "./dataset/loveda"
SAVE_DIR="./checkpoints"

#class name from LOveDA ds
CLASS_NAME =[
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
    "no-data"
]

NUM_CLASSES = 8
IGNORE_INDEX = 7
IN_CHANNEL = 3

#---Device detection---
def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def getDataSet():
    train_ds = LoveDA(root=DATA_PATH, split="train", download=True)
    val_ds= LoveDA(root=DATA_PATH, split="val", download=True)
    test_ds = LoveDA(root=DATA_PATH, split="test", download=True)
    return train_ds, val_ds, test_ds


#---Main---
def main():
    device = getDevice()
    print(f"Device :{device}")

    train_ds, val_ds, test_ds = getDataSet()
    print(f"train : {len(train_ds)} | test : {len(test_ds)} | val : {len(val_ds)}")

    sample = train_ds[0]
    print(f"Image : {sample['image'].shape} dtype : {sample['image'].dtype}")
    print(f"Mask : {sample['mask'].shape} dtype : {sample['mask'].dtype}")
    print(f"Mask value : {sample['mask'].unique().tolist()}")

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    batch = next(iter(train_dl))
    print(f"\n batch image : {batch['image'].shape}")
    print(f"batch mask : {batch['mask'].shape}")



if __name__ == '__main__':
    main()