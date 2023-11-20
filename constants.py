import torch
import pandas as pd
from torchvision.transforms import Compose, ToTensor, Normalize, PILToTensor

CIFAR10_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

CLIP_TRANSFORM = PILToTensor()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_LABELS_TEXT = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# json file from https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json
IMAGENET_LABELS_TEXT = list(pd.read_json("imagenet-simple-labels.json")[0])

if __name__ == "__main__":
    print(IMAGENET_LABELS_TEXT)