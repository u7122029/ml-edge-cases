from .base_pipeline import Base_Pipeline
from .utils import PIL_collate_fn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import DatasetConfig
import torch
from torchvision.transforms import Compose


class CLIP_Pipeline(Base_Pipeline):
    def __init__(self, weights_name, **kwargs):
        self.weights_name = weights_name
        self.processor = CLIPProcessor.from_pretrained(self.weights_name)
        self.model = CLIPModel.from_pretrained(self.weights_name)
        self.vision_model = CLIPVisionModel.from_pretrained(self.weights_name)

        super().__init__(**kwargs)

    def to(self, device):
        super().to(device)
        self.model = self.model.to(self.device)
        self.vision_model = self.vision_model.to(self.device)
        return self

    def __call__(self, dataset_conf: DatasetConfig, label_texts, collate_fn=None, **kwargs):
        if not collate_fn: collate_fn = PIL_collate_fn
        dataset_conf["transform"] = self.pre_image_transform

        dataset = dataset_conf()
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
        output = []
        labels_out = []
        for batch, labels in tqdm(dataloader, total=len(dataloader)):
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)

            inputs = self.processor(text=label_texts, images=batch, return_tensors="pt", padding=True).to(self.device)
            outs = self.model(**inputs).logits_per_image.softmax(dim=1)
            output.append(outs.cpu().detach())
            labels_out.append(labels.cpu().detach())
        return torch.concat(output), torch.concat(labels_out) # return the labels for storage.

    def get_image_features(self, dataset_conf: DatasetConfig, collate_fn=None, **kwargs):
        if not collate_fn: collate_fn = PIL_collate_fn

        dataset_conf["transform"] = self.pre_image_transform
        dataset = dataset_conf()
        dataloader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
        outputs = []
        for batch, _ in tqdm(dataloader, total=len(dataloader)):
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            output = self.model.get_image_features(**inputs)
            outputs.append(output.cpu().detach())
            #outputs.append(output.pooler_output.cpu().detach().flatten())
        outputs = torch.concat(outputs)
        return outputs

    def __str__(self):
        return f"""CLIP_Pipeline"""