import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


#global values
SEED = 33
SNR_LEVELS = [20, 10, 5, 0]
RADIO_SNR_LEVELS = [10, 5]
RADIO_BAND = (300, 3400)
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".acc", ".flac"}

RECORD_DIR = Path(__file__).resolve().parent / "Records"


#---Audio io---
#load, save, find_source, import_recording

def load_audio(path):
    pass

def save_audio(path, signal):
    pass

def find_source(session_dir);
    pass

def import_recording(src, sesssion_dir):
    pass


#---Signal processing---
#radio_effect, add_white_noise

def radio_effect(signal):
    pass

def add_white_noise(voice, snr_db, rng, radio=False):
    pass





#---Add noise for Main---
#add_noise

def add_noise(session_dir):
    
    #load original wav or try to find it
    session_dir = Path(session_dir)
    original = session_dir / "original.wav"
    if not original.exists():
        import_recording(find_source(session_dir), session_dir)

    #load audio and setup noisy folder
    signal = load_audio(original)
    noisy_dir = session_dir / "noisy"
    noisy_dir.mkdir(exist_ok=True)

    #create list with tuple (snr value, is radio)
    conditions = [(snr, False) for snr in SNR_LEVELS] + [(snr, True) for snr in RADIO_SNR_LEVELS]
    log = []

    for snr, radio in conditions:
        pass




#---Dunder secu and tests---

def main():
    #tests go here
    pass

if __name__ == '__main__':
    main()