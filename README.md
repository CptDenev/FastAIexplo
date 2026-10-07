# AI Experiments, Local & Lightweight

> Small models, zero cloud GPU, deployable anywhere.
> Learning architectures by doing, while keeping everything local and optimized.

## Philosophy

- **Local first**: everything trains and runs on personal hardware (CPU or consumer GPU)
- **Small by design**: models < 150 MB, inference < 100 ms, deployable on any device
- **No LLMs for bounded tasks**: a 1.6 MB CNN that reads a digit doesn't need an A100 or prompt engineering
- **Documented decisions**: each project explains why a model was kept, not only how it was trained

## Projects

| Folder | Project | Architecture | Dataset | Size | Status |
|---|---|---|---|---|---|
| [01_fundamentals](01_fundamentals) | MNIST digits, regression basics | MLP, 2×(Conv-Pool) CNN + 2×FC | MNIST | 1.6 MB | ✅ CNN deployed on [HF Space](https://huggingface.co/spaces/CptDenev/MNIST_Digit_CNN) |
| [02_image_classification](02_image_classification) | Room recognition | ResNet18 fine-tuned (fastai) | Photos taken on site | 45 MB | ✅ Deployed on [HF Space](https://huggingface.co/spaces/CptDenev/Conciergerie) |
| [03_predictive_maintenance](03_predictive_maintenance) | Machine failure prediction | MLP vs Random Forest | AI4I 2020 | 2.6 MB | ✅ |
| [04_segmentation](04_segmentation) | Land-cover segmentation | U-Net from scratch vs pretrained encoder | LoveDA | 118 MB | 🚧 Deployment in progress |
| [05_transformer](05_transformer) | Small language model from scratch | Decoder-only transformer | TinyStories | — | 🚧 |
| [06_audio](06_audio) | ASR benchmark on degraded signal | Whisper and Qwen | Clean record signal in French | — | ✅ |

**Next**: synthetic training data generated in Unreal Engine 5 for segmentation, measuring the gap between synthetic and real images (separate repository).

## Stack

- **PyTorch** (CPU, CUDA or MPS depending on the machine)
- **FastAI** for fast iteration
- **Scikit-learn** for classical baselines
- **Gradio** for interactive demos, **HF Spaces** for low-cost hosting
- **Docker** for deployment (first containerized project: segmentation, in progress)

## Design principles

1. **The model must fit on any device** (< 150 MB)
2. **Inference must run on a 2022 laptop** (CPU only is fine)
3. **Deployment should be a single `docker run`**, not a 45-minute setup
4. **Every experiment documents the why**: comments on choices, metrics that can be recomputed
5. **Architectures are explained plainly**

## Repository layout

Datasets live in `dataset/` at the root and are not versioned (except the small AI4I CSV). Scripts resolve paths from the repository root, so they can be launched from anywhere. Checkpoints and figures are written next to each script, in its own `checkpoints/` folder.

---

*Each project = one architecture, one dataset, one documented decision.*