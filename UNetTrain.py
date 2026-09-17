import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

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
UNET_LR = 0.001
UNET_EPOCHS = 50

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


def make_transform(sample, size=512):
    sample['image'] = T.functional.resize(
        sample['image'], size, interpolation=T.InterpolationMode.BILINEAR
    )
    sample['mask'] = T.functional.resize(
        sample['mask'].unsqueeze(0), size, interpolation=T.InterpolationMode.NEAREST
    ).squeeze(0)
    return sample


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
    optimizer = torch.optim.Adam(model.parameters(), lr=UNET_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    scaler = torch.amp.GradScaler()

    best_val = float('inf')

    for epoch in range(1, UNET_EPOCHS+1):
        train_loss = unet_train_one_epoch(model, train_dl, criterion, optimizer, device)
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
    
    unet_train(train_ds,val_ds,device)


if __name__ == '__main__':
    main()