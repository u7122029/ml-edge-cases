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

parser = argparse.ArgumentParser(description="Clusterer")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--image-noun",
    required=False,
    type=str,
    help="The image noun to use for clip models.",
    default="photo"
)
parser.add_argument(
    "--dataset",
    required=False,
    default="cifar10-test",
    help="The name of the dataset that should be used.",
    choices=["imagenet-val", "cifar10-test"]
)
parser.add_argument(
    "--prefix-mod",
    required=False,
    type=str,
    help="The prefix modifier for all ground_labels.",
    default=""
)
parser.add_argument(
    "--suffix-mod",
    required=False,
    type=str,
    help="The suffix modifier for all ground_labels.",
    default=""
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


def decide_transform(dataset_name, model_type, visualise, own_transform=None):
    if own_transform is not None:
        return own_transform

    # Otherwise give default own_transform
    if model_type in CLIP_TRANSFORM_MODELS or visualise:
        return PILToTensor()

    if dataset_name == "cifar10":
        return CIFAR10_TRANSFORM
    elif dataset_name == "imagenet":
        return IMAGENET_TRANSFORM
    else:
        raise Exception(f"Invalid dataset name '{dataset_name}'.")


def get_dataset(dataset_name, split, data_root, model_type, visualise=False, transform=None):
    data_root = Path(data_root)
    transform = decide_transform(dataset_name, model_type, visualise, transform)
    if dataset_name == "cifar10":
        split_map = {"train": True, "test": False}
        return CIFAR10(root=str(data_root), train=split_map[split], transform=transform), CIFAR10_LABELS_TEXT
    elif dataset_name == "imagenet":
        return ImageNet(root=str(data_root / "imagenet"), split=split, transform=transform), IMAGENET_LABELS_TEXT
    else:
        raise Exception(f"Invalid dataset name '{dataset_name}'.")


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


def id(x):
    batch = [a[0] for a in x]
    labels = torch.Tensor([a[1] for a in x])
    return batch, labels


if __name__ == "__main__":
    print(DEVICE)
    args = parser.parse_args()

    data_root = args.data_root
    dataset_full = args.dataset
    dataset_name, dataset_split = dataset_full.split("-")

    model_name = args.model
    model_type, weights_name = model_name_parser(model_name)

    results_path = args.results_path
    image_noun = args.image_noun
    prefix_mod = args.prefix_mod
    suffix_mod = args.suffix_mod

    dset, labels_text = get_dataset(dataset_name, dataset_split, data_root, model_type) #get_dataset("CIFAR10", root="C:/ml_datasets", own_transform=CIFAR10_TRANSFORM, train=False)
    dataloader = DataLoader(dset, batch_size=64, collate_fn=None if model_type not in CLIP_TRANSFORM_MODELS else id)
    pipeline = get_pipeline(model_type, weights_name, dataset_name,
                            label_noun=image_noun,
                            prefix_mod=prefix_mod, suffix_mod=suffix_mod).to(DEVICE)

    file_out = {
        "top10preds": [],
        "top10confs": [],
        "labels": []
    }
    with torch.no_grad():
        for batch, labels in tqdm(dataloader):
            output = pipeline(batch).cpu()
            top10 = torch.topk(output, 10, dim=1)
            confs = top10.values
            preds = top10.indices.to(torch.int16)
            file_out["top10preds"].append(preds)
            file_out["top10confs"].append(confs)
            file_out["labels"].append(labels)

    file_out["top10preds"] = torch.concat(file_out["top10preds"]).to(torch.int16)
    file_out["top10confs"] = torch.concat(file_out["top10confs"])
    file_out["labels"] = torch.concat(file_out["labels"]).to(torch.int16)

    results_file_path = get_output_path(results_path, dataset_full, model_type, weights_name,
                                        image_noun, prefix_mod, suffix_mod)
    results_file_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(file_out, str(results_file_path))