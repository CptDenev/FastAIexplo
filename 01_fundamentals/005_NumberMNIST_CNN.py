import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

#model architecture CNN
#input :    conv1 1 greyscale * (32 * (3*3) filters) + 32 outputs
#hidden :   conv2 (32 in, 64 filters, 3*3 mask), fc1 (64 filters result, 7*7 image size after two maxpool)
#output :   fc2 128 inputs, 10 ouputs (no softmax inside definiton will call on eval)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        #first layer 32 filters created
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        #we keep only the max on 2*2 windows and reduce our spatial dimension (ability to detect better feature)
        self.pool1 = nn.MaxPool2d(2)

        #second layer 64 filters created
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        #reduce again our dimension, both for generalize detection zone and reduce our params
        self.pool2 = nn.MaxPool2d(2)

        #we pass again in 1D
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.relu_fc = nn.ReLU()
        #we keep a random 20% deactivation to avoid over confidence in a specific zone
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x



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


#evaluation function
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
 

    while True:

        print("1: train model on MNSIT")
        print("2: eval report on test set")
        print("any other value : quit")
        choice = int(input("choose selection : "))

        if(choice == 1):
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

                    filename = f"mnist_cnn_epoch{epoch}_loss{best_val_loss:.4f}.pth"
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, filename))
                    print(f"new best val loss = {best_val_loss:.4f}")

                else:
                    patience_counter +=1
                    if patience_counter >= PATIENCE:
                        print(f"early stop at epoch {epoch} we've reached max patience : {PATIENCE}")
                        break

            print("="*60)


            #save the model post training
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, os.path.join( SAVE_DIR,"mnist_cnn_final.pth"))

            #torch.save(model, os.path.join(SAVE_DIR,"mnist_full_model.pth"))


        elif choice == 2 :
            #evaluation on test set
            print("\n" + "="*60)
            print("evaluation on test set")
            print("="*60)

            #load model
            checkpoint = torch.load(
                os.path.join(SAVE_DIR, "mnist_cnn_final.pth"), 
                map_location=device,
                weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])

            test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
            print(f"test loss: {test_loss:.4f}")
            print(f"test accuracy: {test_acc:.4f} ({int(test_acc*10000)}/{len(all_labels)})")

            print("classification report :")
            print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

            print("confusion matrix :")
            print(confusion_matrix(all_labels, all_preds))

            print("prediction example based on softmax :")
            model.eval()
            for i, (images, labels) in enumerate(test_loader):
                if i >= 1:
                    break
                logits = model(images.to(device))
                probs = F.softmax(logits, dim=1)
                for j in range(5):
                    print(f"true : {labels[j].item()} | prediction : {probs[j].argmax().item()} | "
                        f"confidance : {probs[j].max():.4f} | top 3 proba : {probs[j].topk(3)}")

        else :
            break


if __name__ == '__main__':
    main()