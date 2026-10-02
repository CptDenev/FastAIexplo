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

TIME_RE = re.compile()
PRECENT_RE = re.compile()
KM_RE = re.compile()
DIGIT_HYPHEN_RE = re.compile()
DIGIT_LETTER_RE = re.compile()
NUMBER_RE = re.compile()

SNR_ORDER = ["clean", 20, 10, 5, 0]


#---Normalization---

def _time(m):
    pass

def _numbers(m):
    pass

def stip_accents(text):
    pass

def normalize(text):
    pass

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