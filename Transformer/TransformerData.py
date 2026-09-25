import os
import numpy as np

from datasets import load_dataset

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from TransformerConfig import SEED, SPECIAL_TOKENS, DATA_DIR, TOKENIZER_PATH, BOS_TOKEN, EOS_TOKEN, TRAIN_BIN, VAL_BIN, ModelConfig, TrainConfig

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


#tokenize our datas
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


#convert a list of stories into a signel bin file
def encode_split(texts, tokenizer, path):
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    assert tokenizer.get_vocab_size() < 2**16, "vocab to large for uint16"

    #parallel encoding in Rust
    encodings = tokenizer.encode_batch(texts)

    ids=[]
    for enc in encodings:
        ids.append(bos_id)
        ids.extend(enc.ids)
        ids.append(eos_id)

    #we use numpy in place of python list as python list cost 30 octets by int and numpy tab only 2 octets by int
    arr = np.array(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)

    print(f"{len(texts)} stories -> {len(arr)} tokens save at {path}")
    return len(arr)

def get_batch(split, cfg, device):
    pass

#Load saved pipeline
def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"{TOKENIZER_PATH} not found, run prepare_data first")
    return Tokenizer.from_file(TOKENIZER_PATH)

def prepare_data():
    pass

#---Test function---
def check_bin(path, n_tokens, n_stories, tokenizer):
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    #file size must be 2 octets by uint6 token
    size = os.path.getsize(path)
    print(f"{'OK' if size == n_tokens * 2 else 'FAIL'} : file size {size} byte expected {n_tokens*2}")

    #check if start with BOS
    data = np.memmap(path, dtype=np.uint16, mode="r")
    print(f"{'OK' if data[0] == bos_id else 'FAIL'} : first token {data[0]} expected {bos_id}")

    #check if there is only one BOS and one EOS per story
    n_bos = int((data == bos_id).sum())
    n_eos = int((data == eos_id).sum())
    print(f"{'OK' if n_stories == n_bos == n_eos else 'FAIL'} : bos={n_bos} eos={n_eos} expected={n_stories}")

    #visual check, we must see special tokens
    print(repr(tokenizer.decode(data[:100].tolist(), skip_special_tokens=False)))

    #average story length
    print(f"avg tokens per story : {n_tokens / n_stories :.1f}")



#---Sanity check---
def main():
    train, val = load_raw(50_000)
    print(len(train), len(val))
    print(repr(train[0]))

    tok = train_tokenizer(train)
    #vocab size
    vocab_size = tok.get_vocab_size()
    excpeted = ModelConfig().vocab_size
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


    n_train = encode_split(train, tok, TRAIN_BIN)
    check_bin(TRAIN_BIN, n_train, len(train), tok)

    n_val = encode_split(val, tok, VAL_BIN)
    check_bin(VAL_BIN, n_val, len(val), tok)

if __name__ == '__main__':
    main()