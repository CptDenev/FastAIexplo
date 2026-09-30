# Land-cover segmentation: from scratch vs fine-tuned

## Question
Can a U-Net trained from scratch compete with a model built on a pretrained encoder, on a small compute budget?

## Data
[LoveDA](https://github.com/Junjue-Wang/LoveDA): aerial images, 7 classes (background, building, road, water, barren, forest, agriculture). Loaded via TorchGeo.

## Protocol
Identical for both approaches: same split, seed, class weighting and metrics (mIoU and per-class IoU). 6 to 7 runs on each side, the best of each is compared.

## Results

| Model | mIoU | Training time |
|---|---|---|
| U-Net from scratch | [xx] | [xx] |
| Pretrained ResNet encoder, fine-tuned | [xx] | ~10× faster |

[Per-class IoU table and where the gain comes from: to be completed after the post-mortem]

## What changed along the way
- **Phase 1**: hyperparameter tuning on the global metric only.
- **Phase 2**: per-class diagnosis with confusion matrices. This changed the nature of the decisions, from tuning numbers to understanding which classes the models confuse and why.

## Files
- `UNetModel.py`: U-Net architecture
- `UNetTrain.py`: from-scratch training, predictions and confusion matrix export
- `UNetEarlyStop.py`: early stopping
- `UNetFineTune.py`: fine-tuning with a pretrained encoder
- `SegFormerFineTune.py`: transformer-based comparison, next step

## Next
- Deployment of the retained model (Docker, FastAPI, OVHcloud)
- SegFormer in the same pipeline