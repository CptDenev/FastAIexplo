import json
import sys
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
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

PEAK_MAX = 0.99  # peak normalization target, avoids clipping after mixing
SILENCE_TOP_DB = 40  # threshold used to ignore silences when measuring speech power

NATIVE_EXT = {".wav", ".flac"}  # read directly by soundfile, no decoding needed
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".flac"}

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


#Decode compressed audio to float32 WAV at native rate, with the bundled ffmpeg / avoid bad PATH configuration
def decode_to_wav(src, dst):
    
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(src), "-c:a", "pcm_f32le", str(dst),
    ]
    subprocess.run(cmd, check=True)


def import_recording(src, session_dir):
    #check if conversion is needed
    if src.suffix.lower() in NATIVE_EXT:
        signal = load_audio(src)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            decoded = Path(tmp) / "decoded.wav"
            decode_to_wav(src, decoded)
            signal = load_audio(decoded)

    #display peak warning
    peak = float(np.max(np.abs(signal)))
    if peak >= 0.999:
        print(f"warning : {src.name} peaks at {peak:.3f}, the source is probably clipped")

    #save audio file as original.wav
    dst = session_dir / "original.wav"
    save_audio(dst, signal)
    print(f"import {src.name} to {dst.name} ({len(signal) / SAMPLE_RATE:.1f} s, peak {peak:.2f})")
    return dst

#---Signal processing---
#power, speech_power, radio_effect, add_white_noise

#get mean signal power
def power(x):
    return float(np.mean(x ** 2))

#compute power by removing silence which would decreades the mean power value if considered
def speech_power(signal):
    intervals = librosa.effects.split(signal, top_db=SILENCE_TOP_DB)
    voiced = np.concatenate([signal[s:e] for s, e in intervals])
    return power(voiced)

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

    #compute voice power and tune noise according to voice power
    p_voice = speech_power(voice)
    p_noise_target = p_voice / 10 ** (snr_db / 10)
    noise *= np.sqrt(p_noise_target / power(noise))

    #compute SNR and mix voice + noise
    snr_measured = 10 * np.log10(p_voice / power(noise))
    mixed = voice + noise

    #compute peak and normalize gain if peak reach PEAK_MAX
    peak = float(np.max(np.abs(mixed)))
    gain = PEAK_MAX / peak if peak > PEAK_MAX else 1.0

    return (mixed*gain).astype(np.float32), snr_measured, gain


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
        #get randome from SEED for reproductibility
        rng = np.random.default_rng([SEED, snr, int(radio)])
        voice = radio_effect(signal) if radio else signal
        mixed, snr_measured, gain = add_white_noise(voice, snr, rng, radio)

        #save file
        name = f"{'radio_' if radio else ''}snr{snr}.wav"
        save_audio(noisy_dir / name, mixed)

        #logs
        log.append({
            "file": name,
            "snr_target_db": snr,
            "snr_measured_db": round(snr_measured, 2),
            "radio": radio,
            "gain": round(gain, 4),
            "seed": [SEED, snr, int(radio)],
        })
        print(f"{name:<18} target {snr:>3} dB | measured {snr_measured:6.2f} dB | gain {gain:.3f}")

    #get global values
    meta = {
        "sample_rate": SAMPLE_RATE,
        "radio_band_hz": RADIO_BAND,
        "silence_top_db": SILENCE_TOP_DB,
        "conditions": log, 
    }
    #save log file
    (noisy_dir / "conditions.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return log


#---Dunder secu and tests---

def main():
    if len(sys.argv) != 2:
        print("usage: python corpus.py <session_name>")
        sys.exit(1)
    session_dir = RECORD_DIR / sys.argv[1]
    add_noise(session_dir)  # creates original.wav if needed

    signal = load_audio(session_dir / "original.wav")
    intervals = librosa.effects.split(signal, top_db=SILENCE_TOP_DB)
    mask = np.ones(len(signal), dtype=bool)
    for s, e in intervals:
        mask[s:e] = False
    native_snr = 10 * np.log10(speech_power(signal) / power(signal[mask]))
    print(f"native SNR ~ {native_snr:.1f} dB")

if __name__ == '__main__':
    main()