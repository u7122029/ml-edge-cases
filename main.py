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
parser.add_argument("--use-clip",
                    action=argparse.BooleanOptionalAction)
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
    default="cifar10-test",
    help="The name of the dataset that should be used.",
    choices=["imagenet-val", "cifar10-test"]
)
parser.add_argument(
    "--prefix-mod",
    required=False,
    type=str,
    help="The prefix modifier for all labels.",
    default=""
)
parser.add_argument(
    "--suffix-mod",
    required=False,
    type=str,
    help="The suffix modifier for all labels.",
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


def get_dataset(dataset_name, split, data_root, use_clip, visualise=False):
    data_root = Path(data_root)
    if dataset_name == "cifar10":
        transform = PILToTensor() if use_clip or visualise else CIFAR10_TRANSFORM
        split_map = {"train": True, "test": False}
        return CIFAR10(root=str(data_root), train=split_map[split], transform=transform), CIFAR10_LABELS_TEXT
    elif dataset_name == "imagenet":
        transform = PILToTensor() if use_clip or visualise else IMAGENET_TRANSFORM
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
    model_name = args.model
    image_noun = args.image_noun
    dataset_name, split = args.dataset.split("-")
    data_root = args.data_root
    use_clip = args.use_clip
    prefix_mod = args.prefix_mod
    suffix_mod = args.suffix_mod
    results_path = args.results_path

    dset, labels_text = get_dataset(dataset_name, split, data_root, use_clip) #get_dataset("CIFAR10", root="C:/ml_datasets", transform=CIFAR10_TRANSFORM, train=False)
    dataloader = DataLoader(dset, batch_size=64, collate_fn=None if not use_clip else id)
    pipeline = get_pipeline(model_name, dataset_name, use_clip, label_noun=image_noun,
                            prefix_mod=prefix_mod, suffix_mod=suffix_mod).to(DEVICE)

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

    out_path = Path(f"{results_path}/{dataset_name}-{split}")
    out_path.mkdir(parents=True,exist_ok=True)
    noun_repr = f"_{image_noun}" if image_noun and use_clip else ""
    pfmod_repr = f"_{prefix_mod}" if prefix_mod and use_clip else ""
    sfmod_repr = f"_{suffix_mod}" if suffix_mod and use_clip else ""
    torch.save(file_out, str(out_path / f"{model_name}{noun_repr}{pfmod_repr}{sfmod_repr}.pt"))