import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassJaccardIndex

from transformers import SegformerForSemanticSegmentation

import seaborn as sns

from scipy.ndimage import median_filter
from skimage.morphology import closing, disk

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

from UNetEarlyStop import EarlyStopping
from torchgeo.datasets import LoveDA


#---Config---
SEED = 33
DATA_PATH = "./dataset/loveda"
SAVE_DIR = "./checkpoints"

#---Fine tuning config---
SEGFORMER_CKPT = "nvidia/mit-b2"     # ImageNet pretrained MiT encoder (decode head starts from scratch)
ENCODER_TAG = SEGFORMER_CKPT.split("/")[-1].replace("mit-", "")   # "b2"
RUN_TAG = f"segformer_{ENCODER_TAG}_tversky_b4_bld85_brn60"
BATCH_SIZE = 4
FREEZE_EPOCHS = 3            # encoder frozen during the first epochs, decode head adapts alone
ENC_LR = 6e-5                # pretrained encoder, usual SegFormer order of magnitude
DEC_LR = 6e-4                # decode head, trained from scratch
NUM_WORKERS_TRAIN = 8        # lower to 6 / 2 if RAM is tight
NUM_WORKERS_VAL = 4

#class name from LOveDA ds
CLASS_NAME = [
    "no-data",
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture"
]

CLASS_COLORS = [
    "#000000",  # 0 no-data
    "#3C1098",  # 1 background
    "#8429F6",  # 2 building
    "#6EC1E4",  # 3 road
    "#0000FF",  # 4 water
    "#B08C5C",  # 5 barren
    "#228B22",  # 6 forest
    "#FFFF00",  # 7 agriculture
]

cmap = ListedColormap(CLASS_COLORS)
bounds = np.arange(len(CLASS_COLORS) + 1) - 0.5  # [-0.5, 0.5, 1.5, ..., 7.5]
norm = BoundaryNorm(bounds, cmap.N)

NUM_CLASSES = 8
IGNORE_INDEX = 0
UNET_EPOCHS = 150
PATIENCE = 12                # keep aligned with the last resnet34 runs for a fair comparison


def ckpt_path(kind):
    #kind : "best" or "last"
    return f"{SAVE_DIR}/{kind}_{RUN_TAG}.pth"


#---Device detection---
def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def getDataSet():
    #train : augmented / val + test : deterministic (resize only)
    train_ds = LoveDA(root=DATA_PATH, split="train", download=True, transforms=make_transform)
    val_ds = LoveDA(root=DATA_PATH, split="val", download=True, transforms=make_transform_val)
    test_ds = LoveDA(root=DATA_PATH, split="test", download=True, transforms=make_transform_val)

    #train_subset = Subset(train_ds, range(600))
    #val_subset = Subset(val_ds, range(300))

    return train_ds, val_ds, test_ds


#---Model---
class SegformerSeg(nn.Module):
    """
    SegFormer (transformers) with the ImageNet normalisation done INSIDE the model,
    so the pipeline keeps feeding raw 0-255 float images.
    SegFormer outputs logits at 1/4 of the input resolution : they are upsampled
    back to full resolution in forward(), so loss / visu / eval are unchanged.
    """
    def __init__(self, ckpt=SEGFORMER_CKPT, num_classes=8):
        super().__init__()
        id2label = {i: n for i, n in enumerate(CLASS_NAME)}
        label2id = {n: i for i, n in id2label.items()}
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            ckpt,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        #ImageNet mean/std rescaled to the 0-255 range
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.net(pixel_values=(x - self.mean) / self.std).logits   # (B, C, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def get_encoder(model):
    return model.net.segformer


def set_encoder_trainable(model, trainable):
    for p in get_encoder(model).parameters():
        p.requires_grad = trainable


def build_optimizer(model):
    #differential lr : gentle on the pretrained encoder, faster on the decode head
    enc_params = list(get_encoder(model).parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    return torch.optim.AdamW([
        {"params": enc_params, "lr": ENC_LR},
        {"params": other_params, "lr": DEC_LR},
    ], weight_decay=1e-4)


#---Data augmentation---
def make_transform(sample, size=1024):
    image = sample['image']
    mask = sample['mask'].unsqueeze(0)  # (1, H, W)

    # resize
    image = T.functional.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
    mask = T.functional.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)

    #augmentation random flip on H and V
    if torch.rand(1).item() > 0.5:
        image = T.functional.hflip(image)
        mask = T.functional.hflip(mask)
    if torch.rand(1).item() > 0.5:
        image = T.functional.vflip(image)
        mask = T.functional.vflip(mask)

    #augmentation random rot -30 or +30
    angle = int(torch.randint(-30, 30, (1,)).item())
    if angle != 0:
        image = T.functional.rotate(image, angle)
        mask = T.functional.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

    return {'image': image, 'mask': mask.squeeze(0)}


def make_transform_val(sample, size=1024):
    #resize only, no flip / rotation : deterministic validation
    image = sample['image']
    mask = sample['mask'].unsqueeze(0)
    image = T.functional.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
    mask = T.functional.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)
    return {'image': image, 'mask': mask.squeeze(0)}


def postprocess(mask_pred):
    #median filter
    mask_smooth = median_filter(mask_pred, size=3)
    #morphological closing (fill the empty zones)
    mask_final = mask_smooth.copy()

    for c in range(1, NUM_CLASSES):
        binary = (mask_smooth == c).astype(np.uint8)
        #kernel with radius == 2
        closed = closing(binary, disk(radius=2))
        mask_final[closed.astype(bool)] = c

    return mask_final


#---Class weights---
def compute_class_weights(dataset, num_classes, ignore_index, sample_limit=500):
    counts = np.zeros(num_classes)
    for i in range(min(len(dataset), sample_limit)):
        mask = dataset[i]['mask'].numpy()
        for c in range(num_classes):
            if c != ignore_index:
                counts[c] += (mask == c).sum()

    counts[ignore_index] = 1
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights[ignore_index] = 0.0
    weights = weights / weights.sum() * (num_classes - 1)
    return torch.tensor(weights, dtype=torch.float32)


#---DICE loss---
def dice_loss(logits, targets, num_classes, ignore_idx=0, smooth=1e-5):
    #logits : (B, C, H, W)
    #targets : (B, H, W)
    probs = torch.softmax(logits, dim=1)
    #valid mask
    valid = (targets != ignore_idx).float()
    total = 0.0
    n_real_classes = 0

    for c in range(num_classes):
        if c == ignore_idx:
            continue
        pred_c = probs[:, c] * valid
        target_c = ((targets == c) & (targets != ignore_idx)).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice_c = (2 * intersection + smooth) / (union + smooth)
        total += dice_c
        n_real_classes += 1

    return 1 - (total / n_real_classes)


#---Tversky loss---
def tversky_loss(logits, targets, num_classes, ignore_idx=0, alpha=None, beta=None, smooth=1e-5):
    probs = torch.softmax(logits, dim=1)
    valid = (targets != ignore_idx).float()

    # default value : classic dice (0.5/0.5)
    if alpha is None:
        alpha = torch.full((num_classes,), 0.5, device=logits.device)
    if beta is None:
        beta = torch.full((num_classes,), 0.5, device=logits.device)

    total = 0.0
    n_real_classes = 0

    for c in range(num_classes):
        if c == ignore_idx:
            continue
        pred_c = probs[:, c] * valid
        target_c = ((targets == c) & (targets != ignore_idx)).float()

        TP = (pred_c * target_c).sum()
        FP = (pred_c * (1 - target_c)).sum()
        FN = ((1 - pred_c) * target_c * valid).sum()

        tversky_c = (TP + smooth) / (TP + alpha[c] * FP + beta[c] * FN + smooth)
        total += tversky_c
        n_real_classes += 1

    return 1 - (total / n_real_classes)


def get_tversky_weights(num_classes, class_name, device):
    alpha = torch.full((num_classes,), 0.5)
    beta = torch.full((num_classes,), 0.5)

    # building : penalise false positives (bleeding from background)
    idx_building = class_name.index("building")
    alpha[idx_building] = 0.85
    beta[idx_building] = 0.15

    # barren : slightly favour precision (too many false positive blobs)
    idx_barren = class_name.index("barren")
    alpha[idx_barren] = 0.6
    beta[idx_barren] = 0.4

    # road, water : favour recall, let BG appear
    for name in ["road", "water"]:
        idx = class_name.index(name)
        alpha[idx] = 0.4
        beta[idx] = 0.6

    return alpha.to(device), beta.to(device)


#---Combined loss---
def combined_loss(logits, target, num_classes, ignore_idx=0, class_weights=None, alpha=None, beta=None):
    ce = F.cross_entropy(logits, target, ignore_index=ignore_idx, weight=class_weights)
    tv = tversky_loss(logits, target, num_classes, ignore_idx=ignore_idx, alpha=alpha, beta=beta)
    return 0.5 * ce + 0.5 * tv


#---Train one epoch and evaluate---
def seg_train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        with torch.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def seg_evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        mask = batch['mask'].to(device)

        logits = model(images)
        loss = criterion(logits, mask)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def seg_finetune(train_ds, val_ds, device):
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS_TRAIN, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS_VAL, pin_memory=True, persistent_workers=True)

    #model definition
    model = SegformerSeg(SEGFORMER_CKPT, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model : {SEGFORMER_CKPT} | run tag : {RUN_TAG} | params : {n_params/1e6:.1f}M")

    class_weights = compute_class_weights(train_ds, NUM_CLASSES, IGNORE_INDEX).to(device)

    #custom loss with CE + Tversky to maximise "thin" segmentation and punish class bleeding
    tv_alpha, tv_beta = get_tversky_weights(NUM_CLASSES, CLASS_NAME, device)
    criterion = lambda logits, mask: combined_loss(logits, mask, NUM_CLASSES, IGNORE_INDEX, class_weights, tv_alpha, tv_beta)

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    scaler = torch.amp.GradScaler()
    early_stopper = EarlyStopping(patience=PATIENCE)
    best_val = float('inf')

    for epoch in range(1, UNET_EPOCHS + 1):
        #encoder frozen for the first epochs, then unfrozen
        set_encoder_trainable(model, epoch > FREEZE_EPOCHS)
        if epoch == FREEZE_EPOCHS + 1:
            print("--- encoder unfrozen ---")

        train_loss = seg_train_one_epoch(model, train_dl, criterion, optimizer, scaler, device)
        val_loss = seg_evaluate(model, val_dl, criterion, device)
        #we update our LR based on val_loss
        scheduler.step(val_loss)

        print(f"epoch: {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path("best"))

        #early stop
        if early_stopper.step(val_loss):
            print(f"Early stop at {epoch} | best val loss : {early_stopper.best_loss:.4f}")
            break

    # save final aussi
    torch.save(model.state_dict(), ckpt_path("last"))
    print(f"\nSaved ({RUN_TAG}). Best val loss: {best_val:.4f}")


#---visualize predictions---
def visualize_pred(model, dataset, device, n_samples=4):

    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    for i in range(n_samples):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask']

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred = postprocess(pred)

            #logits: (1, 8, H, W)
            #mask:   (1, H, W)
            sample_loss = F.cross_entropy(logits, mask.to(device).unsqueeze(0), ignore_index=IGNORE_INDEX).item()

        #color fix
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        axes[i][0].imshow(img_np / 255.0)
        axes[i][0].set_title("Input")
        axes[i][0].axis('off')

        axes[i][1].imshow(mask.numpy(), cmap=cmap, norm=norm)
        axes[i][1].set_title("Ground truth")
        axes[i][1].axis('off')

        axes[i][2].imshow(pred, cmap=cmap, norm=norm)
        axes[i][2].set_title(f"Pred (loss={sample_loss:.3f})")
        axes[i][2].axis('off')

    legend_elements = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAME[i]) for i in range(len(CLASS_NAME))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/predictions_{RUN_TAG}.png", dpi=100)
    plt.show()


#--- Evaluate confusion---
@torch.no_grad()
def evaluate_confusion(model, loader, num_classes, ignore_index, device):
    model.eval()

    confmat = MulticlassConfusionMatrix(
        num_classes=num_classes, ignore_index=ignore_index
    ).to(device)

    iou_per_class = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=ignore_index, average=None
    ).to(device)

    for batch in loader:
        images = batch["image"].to(device)
        mask = batch["mask"].to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        confmat.update(preds, mask)
        iou_per_class.update(preds, mask)

    return confmat.compute().cpu().numpy(), iou_per_class.compute().cpu().numpy()


def plot_confusion(cm, class_names, ignore_index):
    #delete no-data
    keep = [i for i in range(len(class_names)) if i != ignore_index]
    cm_clean = cm[keep][:, keep]
    labels_clean = [class_names[i] for i in keep]

    #normalization per line
    cm_norm = cm_clean / (cm_clean.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_clean, yticklabels=labels_clean)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized confusion matrix ({RUN_TAG})")
    plt.savefig(f"{SAVE_DIR}/confusion_matrix_{RUN_TAG}.png", dpi=100)
    plt.show()


def load_best_model(device):
    model = SegformerSeg(SEGFORMER_CKPT, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(ckpt_path("best"), map_location=device, weights_only=True))
    return model


#---Main---
def main():
    device = getDevice()
    print(f"Device :{device}")
    print(f"Model : {SEGFORMER_CKPT} | run tag : {RUN_TAG}")

    #get data set and check dtype and length
    train_ds, val_ds, test_ds = getDataSet()
    print(f"train : {len(train_ds)} | test : {len(test_ds)} | val : {len(val_ds)}")

    sample = train_ds[0]
    print(f"Image : {sample['image'].shape} dtype : {sample['image'].dtype}")
    print(f"Mask : {sample['mask'].shape} dtype : {sample['mask'].dtype}")
    print(f"Mask value : {sample['mask'].unique().tolist()}")

    #light loaders only used here for the sanity check and the evaluation
    #(the training loaders with persistent workers are created inside seg_finetune)
    check_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_VAL)

    batch = next(iter(check_dl))
    print(f"\nbatch image : {batch['image'].shape}")
    print(f"batch mask : {batch['mask'].shape}")

    while True:

        print("---Love DA SegFormer fine tuning---")
        print("1: fine tune model")
        print("2: visualize pred on best val loss pth")
        print("3: evaluate confusion on best val loss pth")
        print("0: quit")

        choice = int(input("choose : "))

        if choice == 1:
            #training
            seg_finetune(train_ds, val_ds, device)

        elif choice == 2:
            model = load_best_model(device)
            visualize_pred(model, val_ds, device)

        elif choice == 3:
            model = load_best_model(device)

            cm, ious = evaluate_confusion(model, val_dl, NUM_CLASSES, IGNORE_INDEX, device)

            for i, name in enumerate(CLASS_NAME):
                if i == IGNORE_INDEX:
                    continue
                print(f"{name:12s} IoU : {ious[i]:.3f}")

            plot_confusion(cm, CLASS_NAME, IGNORE_INDEX)

        else:
            break


if __name__ == '__main__':
    main()