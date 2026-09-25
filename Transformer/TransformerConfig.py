import os
from dataclasses import dataclass

SEED = 33

#paths
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

#special tokens
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN]

#data preparation
TRAIN_SUBSET_SIZE = 50_000

@dataclass
class ModelConfig:
    #tokenizer vocab soze
    vocab_size: int = 4096
    #vec dimension
    hidden_size: int = 384
    num_hidden_layers: int = 6
    #attention head with Q,K,V projection
    num_attention_heads: int = 6
    num_key_value_heads: int = 6
    #SwiGLU dim
    intermediate_size: int = 1024
    #max context length
    max_position_embeddings: int = 256
    #rotation freq for RoPE
    rope_theta: float = 10000.0
    #const add in RMSNorm to avoid division by 0
    rms_norm_eps: float = 1e-5
    #share same matrix for embedding and lm_head for output
    tie_word_embeddings: bool = True
    #dropout set to 0
    dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num attention heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num key value heads"
        assert self.head_dim % 2 == 0, \
            f"head_dim ({self.head_dim}) must be even for RoPE"

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


@dataclass
class TrainConfig:
    #training block size
    block_size: int = 256
    #sequence by forward
    batch_size: int = 128
    #forward/back accumulation before optimizer call => token per step 256*64*4 = 65 536
    grad_accum_steps: int = 2
    max_steps: int = 5000
    #linear lr from 0 to lr on 200 first steps
    warmup_steps: int = 200
    lr: float = 5e-4
    min_lr: float = 5e-5
    #regularization for AdamW
    weight_decay: float = 0.1
    #Adam coef => smooth gradient and therefore momentum
    beta1: float = 0.9
    #Adam coef => smooth square and therefore adpatation
    beta2: float = 0.95
    #Adam stability const
    eps: float = 1e-8
    #clip against batches with huge gradient
    grad_clip: float = 1.0
    #step between eval, log and save
    eval_interval: int = 250
    #batch quantity by eval
    eval_iters: int = 50
    #autocast precision
    dtype: str = "bfloat16"
    #how many eval without progress we wait
    early_stop_patience: int = 5


#---Sanity check---
def main():
    ModelConfig()
    print(ModelConfig().head_dim)
    try :
        ModelConfig(hidden_size=100)
        print("FAIL : invalid config accepted")
    except AssertionError as e:
        print(f"OK : {e}")


if __name__ == '__main__':
    main()