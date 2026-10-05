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
    pass



def main():
    if len(sys.argv) not in (3,4):
        print("usage: python run.py <session> <model_name> [cpu|cuda]")
        sys.exit(1)

    device =  sys.argv[3] if len(sys.argv) == 4 else None
    run_session(RECORD_DIR / sys.argv[1], sys.argv[2], device)


if __name__ == '__main__':
    main()