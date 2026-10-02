import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


#global values
SAMPLE_RATE = 16000
SEED = 33
SNR_LEVELS = [20, 10, 5, 0]
RADIO_SNR_LEVELS = [10, 5]
RADIO_BAND = (300, 3400)
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".acc", ".flac"}

RECORD_DIR = Path(__file__).resolve().parent / "Records"


#---Audio io---
#load, save, find_source, import_recording

def load_audio(path):
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return signal.astype(np.float32)

def save_audio(path, signal):
    sf.write(path, signal, SAMPLE_RATE, subtype="PCM_16")


#return the single recording dropped in session folder
def find_source(session_dir):
    #list all audio extension file in folder (name should not be original.wav)
    candidates = [
        p for p in session_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and p.name != "original.wav"
    ]
    #check if there is only one candidate
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected only one recording in {session_dir}, found {len(candidates)}")
    
    return candidates[0]


def import_recording(src, sesssion_dir):
    signal = load_audio(src)
    #compute signal peak
    peak = float(np.max(np.abs(signal)))

    #display peak warning
    if peak >= 0.999:
        print(f"warning : {src.name} peaks at {peak:.3f}, the source is probably clipped")

    dst = sesssion_dir / "original.wav"
    save_audio(dst, signal)
    print(f"import {src.name} to {dst.name} ({len(signal) / SR:.1f} s, peak {peak:.2f})")

    return dst

#---Signal processing---
#radio_effect, add_white_noise


#bandpass filter as defined in RADIO_BAND (300-3400Hz)
def radio_effect(signal):
    sos = butter(4, RADIO_BAND, btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


#add white noise and call radio_effect if radio = True
def add_white_noise(voice, snr_db, rng, radio=False):
    
    #create noise for voice length
    noise = rng.standard_normal(len(voice)).astype(np.float32)
    if radio:
        noise = radio_effect(noise)





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