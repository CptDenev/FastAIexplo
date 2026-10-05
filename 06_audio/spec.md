# 06_audio · Small ASR models under radio-like noise

## Question

What is the smallest speech recognition model that still holds up on a French military-style voice message as noise increases?

The target context is on-premise deployment, where hardware is not guaranteed and smaller models are preferred, so accuracy is always read against model size and latency.

## Corpus

- One French message recorded by me, 30 to 45 seconds, military-style exchange.
- About ten keywords in the message (call sign, numbers, a grid coordinate, a few domain terms).
- `ground_truth.txt` holds the exact transcript, `keywords.txt` one keyword per line.
- Numbers are written in words in the ground truth, and the same rule is applied to every transcript before scoring.

## Noise

- White noise added at a target SNR of 20, 10, 5 and 0 dB. The noise is scaled so that the ratio between signal power and noise power matches the target.
- Radio effect, band-pass 300 to 3400 Hz, applied on top of the 10 dB and 5 dB versions.
- One random generator per condition, seeded with `[SEED, snr, radio]`, so adding a condition later does not change the existing files.
- Peak normalization only if the mix exceeds 0.99, with the same gain on voice and noise, so the SNR is unchanged.
- Measured SNR, gain and seed of every file are logged in `noisy/conditions.json`.
- The original recording is kept as the clean reference.

## Models

| Name | Family | Package | Parameters |
|---|---|---|---|
| whisper-base | Whisper | faster-whisper | 74 M |
| whisper-small | Whisper | faster-whisper | 244 M |
| qwen3-asr-0.6b | Qwen3-ASR | qwen-asr | 0.6 B |
| qwen3-asr-1.7b (optional) | Qwen3-ASR | qwen-asr | 1.7 B |

Every model sits behind the same interface, `transcribe(path) -> text`, so the rest of the code does not know which model runs. Language is forced to French.

Settings, language forced to French for every model. Whisper uses beam search (beam size 5) with VAD disabled, since VAD could cut speech in the noisy conditions and mix two effects in the results. Audio is passed to Whisper as an array rather than a path, which skips its PyAV decoding and keeps decoding out of the measured latency. No int8 quantization in this version.

## Environment

Dedicated virtual environment in `06_audio/.venv`, dependencies pinned in `06_audio/requirements.txt`. `qwen-asr` pins exact versions of `transformers` and `huggingface_hub`, which conflict with the rest of the repo.

## CLI menu

1. Choose a session (a subfolder of `./Records/`).
2. Add noise to the session, imports the raw recording if needed and writes the noisy files into `noisy/`.
3. Choose the model.
4. Run the model on every file of the session, writes one JSON per file into `transcripts/<model>/` (text, latency, real-time factor, device).
5. Score the session, writes `results.csv` and the plots.
0. Quit.

## Metrics

- WER (word error rate) with `jiwer`, after normalization of both reference and transcript (lowercase, no punctuation, numbers in words).
- Keyword recall, share of the keywords from `keywords.txt` found in the transcript.
- Latency per file and real-time factor (processing time divided by audio duration), on GPU and on CPU.
- Model size on disk and peak memory.

## Normalization

Applied identically to the ground truth, the transcripts and the keywords, before any computation.

1. Lowercase.
2. Times, percentages and kilometres expanded ("14h20" becomes "14 heures 20", "60%" becomes "60 pour cent", "2 km" becomes "2 kilometres").
3. Hyphens between digits, and contacts between digits and letters, replaced by spaces.
4. Numbers written in French words with `num2words`. From three digits up, numbers are read digit by digit, following radio procedure for coordinates ("452" becomes "quatre cinq deux").
5. Accents removed.
6. Punctuation, hyphens and apostrophes replaced by spaces.

A rule may only rewrite the same spoken word in another written form. Homophones ("aperçu" and "a perçu", "munition" and "munitions") and near words ("uniforme" for "Uniform") remain errors.

The rules and `keywords.txt` were checked on the clean transcripts of the three base models, then committed before any run on the noisy files. Any later rule goes in a separate commit with its reason, and every model is rescored with it.

## Outputs

- `results.csv`, one row per model and noise condition.
- Plot 1, WER and keyword recall against SNR, one line per model.
- Plot 2, keyword recall against model size, at a fixed noise level.
- A short conclusion in the README, of the form "below X dB, at least model Y is needed".

## Structure

```
06_audio/
  spec.md
  main.py        # CLI menu
  corpus.py      # sessions, ground truth, keywords, noise and radio effect
  models.py      # load_model(name), Whisper and Qwen3-ASR adapters
  run.py         # runs a model on a session, saves transcripts
  metrics.py     # normalization, WER, keyword recall, CSV and plots
  Records/
    <session>/
      <raw recording>        # e.g. .aac from the phone
      original.wav
      ground_truth.txt
      keywords.txt
      noisy/                 # snr20, snr10, snr5, snr0, radio_snr10, radio_snr5, conditions.json
      transcripts/<model>/   # one JSON per audio file
      results.csv
      *.png
```
## Limits

One speaker, one message, one take. The results show a trend on this message, they do not measure general robustness.

## Out of scope (v2)

- A Qwen text model that corrects the raw transcript using the domain vocabulary and extracts a structured JSON (call sign, order, coordinates).
- Context or hotwords given to the ASR model to bias it towards the radio vocabulary.
- Real recorded noise (engine, wind) instead of white noise.
- Streaming transcription.
- int8 quantization.
- Explore newer models like Whistle
