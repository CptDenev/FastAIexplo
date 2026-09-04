# 🧪 AI Experiments — Local & Lightweight

> Small models, zero cloud GPU, deployable anywhere.
> The idea: learn architectures by doing, while keeping everything local and optimized.

## Philosophy

- **Local first** — everything trains and runs on personal hardware (CPU or consumer GPU)
- **Small by design** — models < 50 MB, inference < 100 ms, deployable on any device
- **No LLMs for bounded tasks** — a 1.6 MB CNN that reads a digit doesn't need an A100 or prompt engineering
- **Reproducible** — Dockerfile + seed + config in every project, `docker run` and you're good 🚧

## Projects

| Project | Architecture | Dataset | Size | Status |
|---------|-------------|---------|------|--------|
| MNIST Digit CNN| 2×(Conv-Pool) CNN + 2×FC | MNIST (0-9) | 1.6 MB | ✅ Deployed on [HF Space](https://huggingface.co/spaces/CptDenev/MNIST_Digit_CNN) |
| MLP | Feedforward (3-4 hidden layers) | MNIST (0-9) | 920 kB | ✅ |
| ResNet18 Fine-tune for classification| ResNet18 pretrained (fastai) | photo batch taken on site | 45 Mo | ✅ Deployed on [HF Space] (https://huggingface.co/spaces/CptDenev/Conciergerie) |
| RNN / LSTM — *(upcoming)* | Sequential | — | — | 📋 |
| Attention / small Transformer — *(upcoming)* | Self-attention | — | — | 📋 |

## Stack

- **PyTorch** (CPU, CUDA or MPS depending on the machine)
- **FastAI** for fast iteration developement
- **Gradio** for interactive demos
- **Docker** for reproducible deployment
- **GitHub** for versioning, **HF Spaces** for low-cost hosting

## Design principles

1. **The model must fit on any device** (< 150 MB)
2. **Inference must run on a 2022 laptop** (CPU only is fine)
3. **Final deployment must be a `docker run`**, not a 45-minute setup
4. **Every experiment documents the "why"** — not just code that works, integrate comments on choices and metrics that can be computed
5. **Model Architecture must be explained plainly**

---

*Perpetually under construction. Each project = one architecture, one dataset, one deployment.*