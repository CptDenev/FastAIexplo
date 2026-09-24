import os
import numpy as np

from datasets import load_dataset

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from TransformerConfig import SEED, SPECIAL_TOKENS, DATA_DIR, TOKENIZER_PATH, BOS_TOKEN, EOS_TOKEN, ModelConfig

#load and clean data from TinyStories
def load_raw(subset_size=None):
    ds = load_dataset("roneneldan/TinyStories")
    ds_train = ds["train"]
    ds_val = ds["validation"]

    if subset_size is not None:
        #avoid to ask more stories than effectivly present
        n = min(subset_size, len(ds_train))
        ds_train = ds_train.shuffle(seed=SEED).select(range(n))

    #filter emplty text
    train_texts = [t for t in ds_train["text"] if t.strip()]
    val_texts = [t for t in ds_val["text"] if t.strip()]
    return train_texts, val_texts


def train_tokenizer(texts):
    #load tokenizer model
    tok = Tokenizer(models.BPE())
    #tokenize our text roughly
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    #decoder on bytelevel
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size=ModelConfig().vocab_size, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(DATA_DIR, exist_ok=True)
    tok.save(TOKENIZER_PATH)

    print(f"Tokenizer size : {tok.get_vocab_size()} saved at {TOKENIZER_PATH}")
    return tok

def encode_split(texts, tokenizer, path):
    pass

def get_batch(split, cfg, device):
    pass

def load_tokenizer():
    pass

def prepare_data():
    pass



#---Sanity check---
def main():
    train, val = load_raw(50_000)
    print(len(train), len(val))
    print(repr(train[0]))

    tok = train_tokenizer(train)
    #vocab size
    vocab_size = tok.get_vocab_size()
    excpeted = ModelConfig.vocab_size
    print(f"{'OK' if vocab_size == excpeted else 'FAIL'} : vocab size {vocab_size} (expected {excpeted})")
    
    #special tokens ids
    bos_id = tok.token_to_id(BOS_TOKEN)
    eos_id = tok.token_to_id(EOS_TOKEN)
    print(f"{'OK' if (bos_id, eos_id) == (0,1) else 'FAIL'} : bos={bos_id} eos={eos_id} expected 0 and 1")

    #visual check of tokenization
    enc = tok.encode("Once upon a time")
    print(f"tokens  : {enc.tokens}")
    print(f"ids     : {enc.ids}")

    #test if encode then decode == original text
    ok = all(tok.decode(tok.encode(t).ids) == t for t in train[:100])
    print(f"{'OK' if ok else 'FAIL'} : round trip on 100 stories")

if __name__ == '__main__':
    main()