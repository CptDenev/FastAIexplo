import os
import math
import numpy as np

import torch


#--- Config and hyperparameters ---
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



#--- Main loop ---
def main():
    device = getDevice()


    while True:
        print("1: Train model")
        print("2: Eval best saved model")
        print("3: Generate a story")
        print("4: Export to .gguf")
        print("0: Exit program")
        choice = int(input("Enter a choice : "))




if __name__ == '__main__':
    main()