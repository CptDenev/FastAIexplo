"""
metrics : WER (world error rate), keywrods, plots
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
from num2words import num2words

RECORD_DIR = Path(__file__).resolve().parent / "Records"

#numbers with at least this many digits are read digit by digit (radio procedure for coordinates)
DIGIT_BY_DIGIT_MIN = 3

TIME_RE = re.compile(r"\b(\d{1,2})\s*h(?:\s*(\d{2}))?\b")    # 14h20, 16h, 14 h 20
PERCENT_RE = re.compile(r"(\d+)\s*%")                    # 60%, 60 %
KM_RE = re.compile(r"(\d+)\s*km\b")                      # 2 km, 2km
DIGIT_HYPHEN_RE = re.compile(r"(?<=\d)-(?=\d)")          # 452-871, 4-5-2
DIGIT_LETTER_RE = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")  # 31u -> 31 u
NUMBER_RE = re.compile(r"\d+")

SNR_ORDER = ["clean", 20, 10, 5, 0]
SIZE_CONDITIONS = ["snr10", "snr5", "snr0", "radio_snr10", "radio_snr5"]


#---Normalization---

def _time(m):
    hours = f"{m.group(1)} heures"
    return f"{hours} {m.group(2)}" if m.group(2) else hours

#split in digit by digit uf number are xxx form
def _number(m):
    digits = m.group()
    if len(digits) >= DIGIT_BY_DIGIT_MIN:
        words = " ".join(num2words(int(d), lang="fr") for d in digits)
    else:
        words = num2words(int(digits), lang="fr")
    return f" {words} "

#replace accent by standard character
def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
 
 
def normalize(text):
    """Same rules for reference, hypotheses and keywords. Order matters."""
    #all lower case
    text = text.lower()
    #clean hours to match rule
    text = TIME_RE.sub(_time, text)
    #clean percentage the same way
    text = PERCENT_RE.sub(r"\1 pour cent", text)
    #clean kilometers
    text = KM_RE.sub(r"\1 kilometres", text)
    #replace - by " "
    text = DIGIT_HYPHEN_RE.sub(" ", text)
    #add space if "31u" -> "31 u"
    text = DIGIT_LETTER_RE.sub(" ", text)
    #convert number to text and if number match xxx form => convert it to digit by digit
    text = NUMBER_RE.sub(_number, text)    
    #replace accents by flat letter
    text = strip_accents(text)      
    #repalce all non letters, number or space by a space (- , ')      
    text = re.sub(r"[^a-z0-9]+", " ", text)  
    #clean double or more space and return it
    return " ".join(text.split())

#---Metrics---

#Word Error Count (WER) on normalized string
def wer_detail(ref, hyp):
    if not hyp:
        n = len(ref.split())
        return {"wer": 1.0, "sub": 0, "del": n, "ins": 0}
    out = jiwer.process_words(ref, hyp)
    return {"wer": out.wer, "sub": out.substitutions, "del": out.deletions, "ins": out.insertions}


#One keyword per line, written variants separated by | (first one is the label).
def load_keywords(path):
    keywords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            variants = [v.strip() for v in line.split("|")]
            keywords.append((variants[0], [normalize(v) for v in variants]))
    return keywords

def keyword_hits(hyp, keywords):
    #padding so a keyword only matches whole words
    padded = f" {hyp} "  
    found, missed = [], []
    for label, variants in keywords:
        hit = any(f" {v} " in padded for v in variants)
        (found if hit else missed).append(label)
    return found, missed

#original -> clean, snr10 -> 10, radio_snr5 -> 5 with radio
def parse_condition(stem):
    if stem == "original":
        return "clean", None, False
    radio = stem.startswith("radio_")
    snr = int(stem.removeprefix("radio_").removeprefix("snr"))
    return stem, snr, radio

#---Session scoring---

def score_session(session_dir, plot_snr=5, plot_device="cuda"):
    session_dir = Path(session_dir)
    ref = normalize((session_dir / "ground_truth.txt").read_text(encoding="utf-8"))
    keywords = load_keywords(session_dir / "keywords.txt")
 
    rows = []
    for json_path in sorted((session_dir / "transcripts").glob("*/*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        hyp = normalize(data["text"])
        w = wer_detail(ref, hyp)
        found, missed = keyword_hits(hyp, keywords)
        condition, snr, radio = parse_condition(Path(data["file"]).stem)
        rows.append({
            "model": data["model"],
            "params_m": data.get("params_m"),
            "device": data.get("device"),
            "condition": condition,
            "snr_db": snr,
            "radio": radio,
            "wer": round(w["wer"], 4),
            "sub": w["sub"],
            "del": w["del"],
            "ins": w["ins"],
            "kw_recall": round(len(found) / len(keywords), 4),
            "kw_missed": ";".join(missed),
            "latency_s": data.get("latency_s"),
            "rtf": data.get("rtf"),
        })
 
    if not rows:
        print(f"no transcripts found in {session_dir / 'transcripts'}")
        return []
 
    with open(session_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
 
    for r in rows:
        print(f"{r['model']:<16} {r['device']:<5} {r['condition']:<12} "
              f"WER {r['wer'] * 100:5.1f}% | keywords {r['kw_recall'] * 100:5.1f}%")
 
    # accuracy does not depend on the device, plots use a single one to avoid duplicates
    plot_rows = [r for r in rows if r["device"] == plot_device] or rows
    plot_vs_snr(plot_rows, session_dir / "wer_keywords_vs_snr.png")
    plot_vs_size(plot_rows, session_dir / "keywords_vs_size.png")
    return rows

#---Plots---

def _cond_key(row):
    return "clean" if row["snr_db"] is None else row["snr_db"]
 
#WER and keyword recall vs SNR, one color per model, dashed = radio effect.
def plot_vs_snr(rows, out_path):
    pos = {c: i for i, c in enumerate(SNR_ORDER)}
    models = sorted({r["model"] for r in rows}, key=lambda m: next(
        (r["params_m"] or 0) for r in rows if r["model"] == m))
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, model in enumerate(models):
        for radio, style, suffix in ((False, "-o", ""), (True, "--s", " (radio)")):
            sel = sorted(
                (r for r in rows if r["model"] == model and r["radio"] == radio),
                key=lambda r: pos[_cond_key(r)],
            )
            if not sel:
                continue
            xs = [pos[_cond_key(r)] for r in sel]
            axes[0].plot(xs, [r["wer"] * 100 for r in sel], style, color=f"C{i}", label=model + suffix)
            axes[1].plot(xs, [r["kw_recall"] * 100 for r in sel], style, color=f"C{i}", label=model + suffix)
 
    for ax in axes:
        ax.set_xticks(range(len(SNR_ORDER)))
        ax.set_xticklabels([str(c) for c in SNR_ORDER])
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("WER (%)")
    axes[1].set_ylabel("keyword recall (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path.name}")

 
#keyword recall vs model size, one line per noise condition.
def plot_vs_size(rows, out_path):
    sizes = sorted({(r["params_m"], r["model"]) for r in rows if r["params_m"]})
    if not sizes:
        print("no model sizes, size plot skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cond in enumerate(SIZE_CONDITIONS):
        style = "--s" if cond.startswith("radio") else "-o"
        for j, family in enumerate(("whisper", "qwen3")):
            sel = sorted(
                (r for r in rows if r["condition"] == cond and r["params_m"]
                and r["model"].startswith(family)),
                key=lambda r: r["params_m"],
            )
            if sel:
                ax.plot([r["params_m"] for r in sel], [r["kw_recall"] * 100 for r in sel],
                        style, color=f"C{i}", label=cond if j == 0 else None)

    ax.set_xscale("log")
    ax.set_xticks([p for p, _ in sizes])
    ax.set_xticklabels([m for _, m in sizes], rotation=20, fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("model (log scale of parameters)")
    ax.set_ylabel("keyword recall (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path.name}")

#---Manual check---

#Normalize one raw transcript and show the alignment against the reference.
def check(session_dir, text):
    ref = normalize((session_dir / "ground_truth.txt").read_text(encoding="utf-8"))
    hyp = normalize(text)
    keywords = load_keywords(session_dir / "keywords.txt")

    print(f"REF : {ref}\nHYP : {hyp}\n")
    if hyp:
        print(jiwer.visualize_alignment(jiwer.process_words(ref, hyp)))
    found, missed = keyword_hits(hyp, keywords)
    print(f"keywords {len(found)}/{len(keywords)} | missed: {', '.join(missed) or 'none'}")

#---Dunder secu and tests---
def main():

    if len(sys.argv) == 3 and sys.argv[1] == "score":
        score_session(RECORD_DIR / sys.argv[2])
    
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        check(RECORD_DIR / sys.argv[2], sys.argv[3])
   
    else:
        print('usage: python metrics.py score <session>\n'
              '       python metrics.py check <session> "<raw transcript>"')
        sys.exit(1)

if __name__ == '__main__':
    main()