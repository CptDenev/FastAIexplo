import json
import sys
import time
from pathlib import Path
 
import soundfile as sf
import torch
 
from models import MODELS, default_device, load_model

RECORD_DIR = Path(__file__).resolve().parent / "Records"

#original.wav first, then the noisy files in a stable order.
def session_files(session_dir):
    original = session_dir / "original.wav"
    if not original.exists():
        raise FileNotFoundError(f"{original} not found, run add_noise first")
    noisy = sorted((session_dir / "noisy").glob("*.wav"))
    if not noisy:
        print("warning: no noisy files, only original.wav will be transcribed")
    return [original] + noisy

#device-wide GPU memory in use => warning count other process and not tight to only that script
def gpu_used_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 ** 2


def run_session(session_dir, model_name, device=None):
    #define base variables
    session_dir = Path(session_dir)
    files = session_files(session_dir)
    device = device or default_device()
    out_dir = session_dir / "transcripts" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    #check if gpu available
    baseline_mb = gpu_used_mb() if device == "cuda" else None

    #measure time to load
    t0 = time.perf_counter()
    model = load_model(model_name, device)
    load_s = time.perf_counter() - t0
    print(f"loaded {model_name} on {device} in {load_s:.1f} s")

    #warmup: the first call pays CUDA init and kernel setup, it is not measured
    model.transcribe(files[0])
 
    for path in files:
        audio_s = sf.info(str(path)).duration
 
        t0 = time.perf_counter()
        text = model.transcribe(path)
        latency_s = time.perf_counter() - t0
 
        record = {
            "model": model_name,
            "file": path.name,
            "text": text,
            "device": device,
            "latency_s": round(latency_s, 3),
            "audio_s": round(audio_s, 2),
            "rtf": round(latency_s / audio_s, 4),
            "load_s": round(load_s, 1),
            "params_m": MODELS[model_name]["params_m"],
            # approximate, device-wide; None on CPU for now
            "gpu_mem_mb": round(gpu_used_mb() - baseline_mb) if device == "cuda" else None,
        }
        out_path = out_dir / f"{path.stem}.{device}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name:<18} {latency_s:6.2f} s | RTF {record['rtf']:.3f} | {text[:50]}")
 
    return out_dir



def main():
    if len(sys.argv) not in (3,4):
        print("usage: python run.py <session> <model_name> [cpu|cuda]")
        sys.exit(1)

    device =  sys.argv[3] if len(sys.argv) == 4 else None
    run_session(RECORD_DIR / sys.argv[1], sys.argv[2], device)


if __name__ == '__main__':
    main()