from torchvision.datasets import CIFAR10, ImageNet
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.transforms import PILToTensor
from constants import *
from pathlib import Path
from models import get_pipeline
import argparse

parser = argparse.ArgumentParser(description="Pretrained Model Tester")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run." # TODO: ADD VALID CHOICES
)
parser.add_argument(
    "--image-noun",
    required=False,
    type=str,
    help="The image noun to use for clip models.",
    default=None
)
parser.add_argument(
    "--dataset",
    required=False,
    default="cifar10",
    help="The name of the dataset that should be used.",
    choices=["imagenet-val", "cifar10-test"]
)
parser.add_argument(
    "--data-root",
    required=False,
    default="C:/ml_datasets",
    type=str,
    help="path containing all datasets (training and validation)"
)
parser.add_argument(
    "--results-path",
    required=False,
    default="results",
    type=str,
    help="The path to store results."
)


def get_dataset(dsetname, *args, **kwargs):
    if dsetname == "CIFAR10":
        return CIFAR10(*args, **kwargs), CIFAR10_LABELS_TEXT
    elif dsetname == "IMAGENET":
        return ImageNet(*args, **kwargs), IMAGENET_LABELS_TEXT
    else:
        raise Exception(f"Invalid dataset name '{dsetname}'.")

def a_or_an(word):
    vowels = "aeiouy"
    if word[0] in vowels:
        return "an"
    return "a"


def generate_text_lst(classes, image_noun):
    if not image_noun: return classes
    out = []
    for c in classes:
        out.append(f"{a_or_an(image_noun)} {image_noun} of {a_or_an(c)} {c}")
    return out


if __name__ == "__main__":
    print(DEVICE)
    args = parser.parse_args()
    model_name = args.model
    image_noun = args.image_noun
    dataset_name = args.dataset
    data_root = args.data_root
    dset, labels_text = get_dataset(dataset_name, model_name, data_root) #get_dataset("CIFAR10", root="C:/ml_datasets", transform=CIFAR10_TRANSFORM, train=False)
    dataloader = DataLoader(dset, batch_size=64)
    pipeline = get_pipeline(model_name, "CIFAR10", label_noun=image_noun).to(DEVICE)

    file_out = {
        "top3preds": [],
        "top3confs": [],
        "labels": []
    }
    with torch.no_grad():
        for batch, labels in tqdm(dataloader):
            output = pipeline(batch).cpu()
            top3 = torch.topk(output, 3, dim=1)
            confs = top3.values
            preds = top3.indices.to(torch.int16)
            file_out["top3preds"].append(preds)
            file_out["top3confs"].append(confs)
            file_out["labels"].append(labels)

    file_out["top3preds"] = torch.concat(file_out["top3preds"]).to(torch.int16)
    file_out["top3confs"] = torch.concat(file_out["top3confs"])
    file_out["labels"] = torch.concat(file_out["labels"]).to(torch.int16)

    out_path = Path("results/cifar10-test")
    out_path.mkdir(parents=True,exist_ok=True)
    torch.save(file_out, str(out_path / f"{model_name}.pt"))