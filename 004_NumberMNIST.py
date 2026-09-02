import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

#model architecture
#input :    28*28 (784px flatten)
#hidden :   256 -> 128
#output :   10 (no softmax inside definiton will call on eval)
#dropout to disable 20% random neuron on each epoch and avoid excessive confidence or false patern detection
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128,10)

        )

    def forward(self, x):
        return self.net(x)





#training function
def train_one_epoch(model, loader, criterion, optimizer, device):
    #put model in train mode
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        #load images and labels to device
        images, labels = images.to(device), labels.to(device)

        #forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #compute stats
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    mean_loss = total_loss / total
    correct_pred = correct / total
    return mean_loss , correct_pred


#evaluation
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    #put model in eval mode
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        #get max logits for tensor, deplace it to RAM to prevent crash from numpy and convert it to np array
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        #we use extend and not append in order to add element one by one
        all_labels.extend(labels.cpu().numpy())

    #compar pred with correct labels and mean the result
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    mean_loss = total_loss / len(loader.dataset)

    return mean_loss, acc, all_preds, all_labels


def main():

    #Base setup and variables
    SEED = 33
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    #check cuda or mps availability, else cpu
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
    elif(torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print(f"device on which we're running: {device}")

    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-3
    PATIENCE = 5
    DATA_DIR = "./dataset/mnist"
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)


    #Load MNIST and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform) 
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    print(f"train set : {len(train_dataset)} images")
    print(f"test set : {len(test_dataset)} images")


    #Split train in train + eval
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator = torch.Generator().manual_seed(SEED)
    )

    print(f"train set : {len(train_subset)} | validation set : {len(val_subset)} | test set {len(test_dataset)}")



    #define data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)




    #create model, load it to device and define criterion, opti and scheduler
    model = MNISTNet().to(device)
    print(f"parameters : {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)




    #training loop with validation and early stop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}

    print("\n" + "="*60)
    print(f"{'epoch':<6}{'train loss':<12}{'train acc':<12}{'val loss':<12}{'val acc':<12}")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:<6}{train_loss:<12.4f}{train_acc:<12.4f}{val_loss:<12.4f}{val_acc:<12.4f}")

        #early stop test
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            patience_counter = 0

            filename = f"mnist_epoch{epoch}_loss{best_val_loss:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, filename))
            print(f"new best val loss = {best_val_loss:.4f}")

        else:
            patience_counter +=1
            if patience_counter >= PATIENCE:
                print(f"early stop at epoch {epoch} we've reached max patience : {PATIENCE}")
                break

    print("="*60)


    #save the model

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "epcoh": epoch,
    }, os.path.join( SAVE_DIR,"mnist_final.pth"))

    #torch.save(model, os.path.join(SAVE_DIR,"mnist_full_model.pth"))



if __name__ == '__main__':
    main()