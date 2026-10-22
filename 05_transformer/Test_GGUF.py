import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from TransformerConfig import TOKENIZER_PATH, BOS_TOKEN, EOS_TOKEN, BASE_DIR

OUT_DIR = os.path.join(BASE_DIR, "hf_test")

def main():
    #wrap our tokenizer.json into a HF tokenizer, declaring special tokens
    tok = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    print(f"vocab {len(tok)} | bos={tok.bos_token_id} eos={tok.eos_token_id}")

    #tiny Llama, random weights : debug preset sizes
    config = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params : {n_params:,}")

    #HF side sanity : generation runs end to end (output is garbage)
    ids = tok("Once upon a time", return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=20, do_sample=False)
    print(repr(tok.decode(out[0], skip_special_tokens=False)))

    #save in HF format : config.json, model.safetensors, tokenizer files
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(sorted(os.listdir(OUT_DIR)))

if __name__ == "__main__":
    main()