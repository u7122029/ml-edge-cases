import torch
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