import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
import matplotlib.pyplot as plt

from UNetModel import UNet
from torchgeo.datasets import LoveDA


#---Config---
SEED = 33
DATA_PATH = "./dataset/loveda"
SAVE_DIR="./checkpoints"

#class name from LOveDA ds
CLASS_NAME =[
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
    "no-data"
]

NUM_CLASSES = 8
IGNORE_INDEX = 7
IN_CHANNEL = 3
UNET_LR = 3e-4
UNET_EPOCHS = 150

#---Device detection---
def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def getDataSet():
    train_ds = LoveDA(root=DATA_PATH, split="train", download=True, transforms=make_transform)
    val_ds= LoveDA(root=DATA_PATH, split="val", download=True, transforms=make_transform)
    test_ds = LoveDA(root=DATA_PATH, split="test", download=True, transforms=make_transform)

    train_subset = Subset(train_ds, range(600))
    val_subset = Subset(val_ds, range(300))

    return train_subset, val_subset, test_ds

#---Data augmentation---
def make_transform(sample, size=768):
    image = sample['image']
    mask  = sample['mask'].unsqueeze(0)  # (1, H, W)

    # resize
    image = T.functional.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
    mask  = T.functional.resize(mask, size, interpolation=T.InterpolationMode.NEAREST)

    # augmentations
    if torch.rand(1).item() > 0.5:
        image = T.functional.hflip(image)
        mask  = T.functional.hflip(mask)
    if torch.rand(1).item() > 0.5:
        image = T.functional.vflip(image)
        mask  = T.functional.vflip(mask)

    angle = int(torch.randint(-30, 30, (1,)).item())
    if angle != 0:
        image = T.functional.rotate(image, angle)
        mask  = T.functional.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

    return {'image': image, 'mask': mask.squeeze(0)} 






def unet_train_one_epoch(model, loader, criterion, optimizer, scaler, device):
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
def unet_evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        images = batch['image'].to(device)
        mask = batch['mask'].to(device)

        logits = model(images)
        loss = criterion(logits, mask)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def unet_train(train_ds, val_ds, device):
    torch.manual_seed(SEED)

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    #model definition
    model = UNet(in_channels=IN_CHANNEL, num_classes=NUM_CLASSES, base_filters=64).to(device)
    #loss function with ignore pixel given
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=UNET_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    scaler = torch.amp.GradScaler()

    best_val = float('inf')

    for epoch in range(1, UNET_EPOCHS+1):
        train_loss = unet_train_one_epoch(model, train_dl, criterion, optimizer, scaler, device)
        val_loss =  unet_evaluate(model, val_dl, criterion, device)
        #we update our LR based on val_loss
        scheduler.step(val_loss)

        print(f"epoch: {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_unet.pth")

    # save final aussi
    torch.save(model.state_dict(), f"{SAVE_DIR}/last_unet.pth")
    print(f"\nSaved. Best val loss: {best_val:.4f}")


def visualize_pred(model, dataset, device, n_samples=4):

    model.eval()
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))

    for i in range(n_samples):
        sample = dataset[i]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask']

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1).squeeze(0).cpu()
            #logits: (1, 8, 768, 768)
            #mask:   (1, 768, 768)
            sample_loss = F.cross_entropy(logits, mask.to(device).unsqueeze(0), ignore_index=7).item()

        #color fix
        img_np = image[0].permute(1,2,0).cpu().numpy()
        axes[i][0].imshow(img_np/255.0)
        axes[i][0].set_title("Input")
        axes[i][0].axis('off')

        axes[i][1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=7)
        axes[i][1].set_title("Ground truth")
        axes[i][1].axis('off')

        axes[i][2].imshow(pred.numpy(), cmap='tab10', vmin=0, vmax=7)
        axes[i][2].set_title(f"Pred (loss={sample_loss:.3f})")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.savefig("./checkpoints/predictions", dpi=100)
    plt.show()


#---Main---
def main():
    device = getDevice()
    print(f"Device :{device}")

    #get data set and check dtype and length
    train_ds, val_ds, test_ds = getDataSet()
    print(f"train : {len(train_ds)} | test : {len(test_ds)} | val : {len(val_ds)}")

    sample = train_ds[0]
    print(f"Image : {sample['image'].shape} dtype : {sample['image'].dtype}")
    print(f"Mask : {sample['mask'].shape} dtype : {sample['mask'].dtype}")
    print(f"Mask value : {sample['mask'].unique().tolist()}")

    
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    batch = next(iter(train_dl))
    print(f"\nbatch image : {batch['image'].shape}")
    print(f"batch mask : {batch['mask'].shape}")

    #training
    #unet_train(train_ds,val_ds,device)

    #unit test
    model = UNet(in_channels=IN_CHANNEL, num_classes=NUM_CLASSES, base_filters=64).to(device)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_unet.pth", map_location=device, weights_only=True))
    visualize_pred(model, val_ds,device)

if __name__ == '__main__':
    main()