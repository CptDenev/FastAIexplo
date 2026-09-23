from dataclasses import dataclass

@dataclass
class ModelCongif:
    #tokenizer vocab soze
    vocab_size: int = 4096
    #vec dimension
    hidden_size: int = 384
    num_hidden_layer: int = 6
    #attention head with Q,K,V projection
    num_attention_head: int = 6
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


@dataclass
class TrainConfig:
    seed: int = 33
    #training block size
    block_size: int = 256
    #sequence by forward
    batch_size: int = 64
    #forward/back accumulation before optimizer call => token per step 256*64*4 = 65 536
    grad_accum_steps: int = 4
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