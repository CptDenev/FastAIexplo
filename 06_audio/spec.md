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
- Fixed seed, so regenerating the noisy files gives identical results.
- The original recording is kept as the clean reference.

## Models

| Name | Family | Package |
|---|---|---|
| whisper-base | Whisper | faster-whisper |
| whisper-small | Whisper | faster-whisper |
| qwen3-asr-0.6b | Qwen3-ASR | qwen-asr |
| qwen3-asr-1.7b (optional) | Qwen3-ASR | qwen-asr |

Every model sits behind the same interface, `transcribe(path) -> text`, so the rest of the code does not know which model runs. Language is forced to French.

## CLI menu

1. Add noise to a session (`./Records/<session>/`), writes the noisy files into `noisy/`.
2. Choose the model.
3. Run the model on every file of a session, writes one JSON per file into `transcripts/<model>/` (text, latency, device).
4. Score a session, writes `results.csv` and the plots.
0. Quit.

## Metrics

- WER (word error rate) with `jiwer`, after normalization of both reference and transcript (lowercase, no punctuation, numbers in words).
- Keyword recall, share of the keywords from `keywords.txt` found in the transcript.
- Latency per file and real-time factor (processing time divided by audio duration), on GPU and on CPU.
- Model size on disk and peak memory.

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
      original.wav
      ground_truth.txt
      keywords.txt
      noisy/
      transcripts/<model>/
      results.csv
```

## Limits

One speaker, one message, one take. The results show a trend on this message, they do not measure general robustness.

## Out of scope (v2)

- A Qwen text model that corrects the raw transcript using the domain vocabulary and extracts a structured JSON (call sign, order, coordinates).
- Real recorded noise (engine, wind) instead of white noise.
- Streaming transcription.
- int8 quantization.
