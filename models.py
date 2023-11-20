import torchvision.models as tvm
import torch
from abc import ABC, abstractmethod
from transformers import CLIPProcessor, CLIPModel
from constants import *

class Model_Pipeline(ABC):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.model = self.model.to(self.device)
        return self

    @abstractmethod
    def process_input(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def process_output(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, batch):
        """
        Idea: process input, feed into self.model, extract file_out from model results.
        """
        pass


class Classic_Pipeline(Model_Pipeline):
    def __init__(self, model):
        super().__init__(model)

    def process_input(self, x):
        return x.to(self.device)

    def forward(self, input):
        output = self.model(input)
        return output

    def process_output(self, forward_output):
        return torch.softmax(forward_output, dim=1)

    def __call__(self, batch):
        processed_input = self.process_input(batch)
        output = self.forward(processed_input)
        processed_output = self.process_output(output)
        return processed_output


class CLIP_Pipeline(Model_Pipeline):
    def __init__(self, model, label_texts):
        super().__init__(model)
        self.label_texts = label_texts
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process_input(self, images):
        return self.processor(text=self.label_texts, images=images,
                                return_tensors="pt", padding=True).to(self.device)

    def forward(self, inputs):
        return self.model(**inputs)

    def process_output(self, outputs):
        return outputs.logits_per_image.softmax(dim=1)

    def __call__(self, batch):
        processed_input = self.process_input(batch)
        output = self.forward(processed_input)
        processed_output = self.process_output(output)
        return processed_output


def get_model_cifar10(name):
    pytorch_cifar_models = set([f"resnet{i}" for i in [20,32,44,56]] + ["mobilenetv2_x1_4"])
    if name in pytorch_cifar_models:
        return torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{name}",
                              pretrained=True, trust_repo=True)
    elif name == "clip":
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise Exception(f"Unknown model {name}.")


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


def get_label_text_lst(dataset, extensions=None, image_noun=None):
    if dataset == "CIFAR10":
        text_lst = CIFAR10_LABELS_TEXT
    elif dataset == "IMAGENET":
        text_lst = IMAGENET_LABELS_TEXT
    else:
        raise Exception("Unknown dataset.")

    text_lst += extensions if extensions else []
    return generate_text_lst(text_lst, image_noun)


def get_model(name, dataset):
    if dataset == "CIFAR10":
        return get_model_cifar10(name)
    elif dataset == "IMAGENET":
        return tvm.get_model(name, weights="DEFAULT")
    else:
        raise Exception("Invalid dataset")


def get_pipeline(name, dataset, extensions=None, label_noun=None):
    model = get_model(name, dataset)
    if isinstance(model, CLIPModel):
        label_texts = get_label_text_lst(dataset, extensions, label_noun)
        return CLIP_Pipeline(model, label_texts)
    else:
        return Classic_Pipeline(model)


if __name__ == "__main__":
    pipeline = get_pipeline("clip", "CIFAR10")
    print(pipeline)