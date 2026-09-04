from torchvision import datasets, transforms
from PIL import Image
import os

# Télécharge le test set MNIST (si pas déjà local)
ds = datasets.MNIST(root="./mnist_data", train=False, download=True)

os.makedirs("test_mnist", exist_ok=True)
for i in range(10):
    img, label = ds[i]  # PIL Image 28x28 grayscale
    img.save(f"test_mnist/sample_{label}.png")
    print(f"sample_{label}.png → digit {label}")