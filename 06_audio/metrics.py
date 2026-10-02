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

TIME_RE = re.compile(r"\b(\d{1,2})\s*h\s*(\d{2})?\b")    # 14h20, 16h, 14 h 20
PERCENT_RE = re.compile(r"(\d+)\s*%")                    # 60%, 60 %
KM_RE = re.compile(r"(\d+)\s*km\b")                      # 2 km, 2km
DIGIT_HYPHEN_RE = re.compile(r"(?<=\d)-(?=\d)")          # 452-871, 4-5-2
DIGIT_LETTER_RE = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")  # 31u -> 31 u
NUMBER_RE = re.compile(r"\d+")

SNR_ORDER = ["clean", 20, 10, 5, 0]


#---Normalization---

def _time(m):
    hours = f"{m.group(1)} heures"
    return f"{hours} {m.group(2)}" if m.group(2) else hours

def _number(m):
    digits = m.group()
    if len(digits) >= DIGIT_BY_DIGIT_MIN:
        words = " ".join(num2words(int(d), lang="fr") for d in digits)
    else:
        words = num2words(int(digits), lang="fr")
    return f" {words} "

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

def wer_details(ref, hyp):
    pass

def load_keaywords(path):
    pass

def keyword_hits(hyp, keywords):
    pass

def parse_condition(stem):
    pass

#---Session scoring---

def score_session(session_dir, plot_snr=5, plot_device="cuda"):
    pass

#---Plots---

def _cond_key(row):
    pass

def plot_vs_snr(rows, out_path):
    pass

def plot_vs_size(rows, snr, out_path):
    pass

#---Manual check---
def check(session_dir, text):
    pass

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