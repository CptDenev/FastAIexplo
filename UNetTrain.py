import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from UNetModel import UNet

#---Device detection---
def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")





#---Main---
def main():
    device = getDevice()


if __name__ == '__main__':
    main()