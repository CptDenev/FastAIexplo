import os
import math
import numpy as np

import torch


#--- Config ---
SEED = 33
DATA_PATH = "./data"
SAVE_PATH = "./checkpoints"
TOKENIZER_PATH = "./tokenizer"

VOCAB_SIZE = 8192

BATCH_SIZE = 32



#--- Device detection ---
def getDevice():
    if torch.cuda.is_available():
        torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else :
        return torch.device("cpu")


