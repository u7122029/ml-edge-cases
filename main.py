from torchvision.datasets import CIFAR10
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from constants import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ToNumpyArray:
    """
    Takes a PIL image and converts it into a numpy array.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


def get_dataset(dsetname, *args, **kwargs):
    if dsetname == "CIFAR10":
        return CIFAR10(*args, **kwargs)
    else:
        raise Exception(f"Invalid dataset name '{dsetname}'.")


def a_or_an(word):
    vowels = "aeiouy"
    if word[0] in vowels:
        return "an"
    return "a"


def generate_text_lst(classes, image_noun):
    out = []
    for c in classes:
        out.append(f"{a_or_an(image_noun)} {image_noun} of {a_or_an(c)} {c}")
    return out


if __name__ == "__main__":
    print(DEVICE)
    dset = get_dataset("CIFAR10", root="C:/ml_datasets", transform=ToNumpyArray(), train=False)

    output = dict(top3preds=[], top3confs=[])
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    texts = generate_text_lst(CIFAR10_LABELS_TEXT, "photo")
    with torch.no_grad():
        for image, label in tqdm(dset):
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(DEVICE)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image.cpu().flatten()
            top3 = torch.topk(logits_per_image.softmax(0),3)
            confs = top3.values.detach().numpy()
            preds = top3.indices.to(torch.uint8).detach().numpy()
            output["top3preds"].append(preds)
            output["top3confs"].append(confs)

    output["top3preds"] = torch.Tensor(np.stack(output["top3preds"]))
    output["top3confs"] = torch.Tensor(np.stack(output["top3confs"]))

    torch.save(output, "results/cifar10-test/clip.pt")