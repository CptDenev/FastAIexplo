import json
import sys
import time
from pathlib import Path
 
import soundfile as sf
import torch
 
from models import MODELS, default_device, load_model

RECORD_DIR = Path(__file__).resolve().parent / "Records"


def session_files(session_dir):
    pass

def gpu_used_mb():
    pass

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