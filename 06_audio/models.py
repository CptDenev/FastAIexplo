"""
models: adapter and model registery
"""

import sys
import time

import torch
import soundfile as sf

#---Device check---
def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#---Adaptaters---

class WhisperAdapter:
    def __init__(self, model_id, device):
        from faster_whisper import WhisperModel

        self.device = device
        compute_type = "float16" if device == "cuda" else "float32"
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(self, path):
        audio, sr = sf.read(str(path), dtype="float32")
        if sr != 16000:
            raise ValueError(f"{path} is {sr} Hz, expected 16000")
        segments, _ = self.model.transcribe(
            audio,
            language="fr",
            beam_size=5,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments)


class QwenASRAdapter:
    def __init__(self, model_id, device):
        from qwen_asr import Qwen3ASRModel

        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.bfloat32
        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype = dtype,
            device_map = "cuda:0" if device == "cuda" else "cpu",
            max_new_tokens = 512
        )

    def transcribe(self, path):
        results = self.model.transcribe(audio=str(path), language="French")
        return results[0].text



#---Model registery---
MODELS = {
    "whisper-base": {"adapter": WhisperAdapter, "model_id": "base", "params_m": 74},
    "whisper-small": {"adapter": WhisperAdapter, "model_id": "small", "params_m": 244},
    "qwen3-asr-0.6b": {"adapter": QwenASRAdapter, "model_id": "Qwen/Qwen3-ASR-0.6B", "params_m": 600},
    "qwen3-asr-1.7b": {"adapter": QwenASRAdapter, "model_id": "Qwen/Qwen3-ASR-1.7B", "params_m": 1700},
}

def load_model(name, device = None):
    if name not in MODELS:
        raise KeyError(f"unknow model {name}, available: {list(MODELS)}")

    entry = MODELS[name]
    device = device or default_device()
    model = entry["adapter"](entry["model_id"], device)
    model.name = name
    return model



#---Dunder secu and tests---

def main():
    if len(sys.argv) not in (3, 4):
        print("usage: python models.py <model_name> <wav_path> [cpu|cuda]")
        sys.exit(1)
    device = sys.argv[3] if len(sys.argv) == 4 else None
 
    t0 = time.perf_counter()
    model = load_model(sys.argv[1], device)
    print(f"loaded {model.name} on {model.device} in {time.perf_counter() - t0:.1f} s")
 
    t0 = time.perf_counter()
    text = model.transcribe(sys.argv[2])
    print(f"transcribed in {time.perf_counter() - t0:.2f} s (cold, includes warmup)")
    print(text)


if __name__ == '__main__':
    main()