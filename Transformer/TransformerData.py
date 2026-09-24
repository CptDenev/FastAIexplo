import os
import numpy as np

from datasets import load_dataset

from TransformerConfig import SEED

def load_raw(subset_size=None):
    ds = load_dataset("roneneldan/TinyStories")
    ds_train = ds["train"]
    ds_val = ds["validation"]

    if subset_size is not None:
        #avoid to ask more stories than effectivly present
        n = min(subset_size, len(ds_train))
        ds_train = ds_train.shuffle(seed=seed).select(range(n))

    #filter emplty text
    train_texts = [t for t in ds_train["text"] if t.strip()]
    val_texts = [t for t in ds_val["text"] if t.strip()]
    return ds_train, ds_val

def train_tokenizer(texts):
    pass

def encode_split(texts, tokenizer, path):
    pass


def get_batch(split, cfg, device):
    pass

def load_tokenizer():
    pass

def prepare_data():
    pass



#TEST
def main():
    train, val = load_raw(50_000)
    print(len(train), len(val))
    print(repr(train[0]))


if __name__ == '__main__':
    main()