import torch
import pandas as pd
from torchvision.transforms import (Compose,
                                    ToTensor,
                                    Normalize,
                                    PILToTensor,
                                    Resize,
                                    CenterCrop)
from pathlib import Path

CIFAR10_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
IMAGENET_TRANSFORM = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
CLIP_TRANSFORM = PILToTensor()
CLIP_TRANSFORM_MODELS = {"clip", "blip", "altclip", "groupvit", "owlvit"}

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

DATA_PATH_DEFAULT = "C:/ml_datasets"
FIGURES_PATH_DEFAULT = "figures"

VALID_MODELS = [
    "clip@openai/clip-vit-large-patch14",
    "clip@openai/clip-vit-base-patch16",
    "clip@openai/clip-vit-base-patch32",
    "clip@openai/clip-vit-large-patch14-336",
    "clip@laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "clip@laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "clip@laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "clip@laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "altclip@BAAI/AltCLIP",
    "altclip@BAAI/AltCLIP-m9",
    "altclip@BAAI/AltCLIP-m18",
    "groupvit@nvidia/groupvit-gcc-yfcc",
    "groupvit@nvidia/groupvit-gcc-redcaps",
    "owlvit@google/owlvit-base-patch32",
    "owlvit@google/owlvit-base-patch16",
    "owlvit@google/owlvit-large-patch14",
    "blip@Salesforce/blip-itm-base-coco",
    "blip@Salesforce/blip-itm-large-coco",
    "blip@Salesforce/blip-itm-base-flickr"
]


def model_name_parser(inp):
    return inp.split("@")


def get_output_path(results_dir, dataset_name, model_type, weights_name, image_noun, prefix_mod, suffix_mod,
                    filetype="pt"):
    s_image_noun = f"_{image_noun}" if image_noun else ""
    s_prefix_mod = f"_{prefix_mod}" if prefix_mod else ""
    s_suffix_mod = f"_{suffix_mod}" if suffix_mod else ""
    return (Path(results_dir) / dataset_name / model_type /
            f"{weights_name}{s_image_noun}{s_prefix_mod}{s_suffix_mod}.{filetype}")


if __name__ == "__main__":
    print(IMAGENET_LABELS_TEXT)