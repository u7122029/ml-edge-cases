from .base_pipeline import Model_Pipeline
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from tqdm import tqdm
import torch


class CLIP_Pipeline(Model_Pipeline):
    def __init__(self, weights_name, label_texts, device="cpu"):
        super().__init__("clip", weights_name, device)
        self.label_texts = label_texts

    def _Model_Pipeline__get_model(self):
        self.processor = CLIPProcessor.from_pretrained(self.weights_name)
        self.model = CLIPModel.from_pretrained(self.weights_name)
        self.vision_model = CLIPVisionModel.from_pretrained(self.weights_name)

    def _Model_Pipeline__put_on_device(self, device):
        super()._Model_Pipeline__put_on_device(device)
        self.model = self.model.to(self.device)
        self.vision_model = self.vision_model.to(self.device)

    def process_input(self, images):
        return self.processor(text=self.label_texts, images=images,
                              return_tensors="pt", padding=True).to(self.device)

    def forward(self, inputs):
        return self.model(**inputs)

    def process_output(self, outputs):
        return outputs.logits_per_image.softmax(dim=1)

    def get_image_features(self, batch):
        outputs = []
        for image in tqdm(batch, total=len(batch)):
            input = self.processor(images=[image], return_tensors="pt").to(self.device)
            output = self.vision_model(**input)
            outputs.append(output.pooler_output.cpu().detach().flatten())
        outputs = torch.stack(outputs)
        return outputs

    def __call__(self, batch):
        processed_input = self.process_input(batch)
        output = self.forward(processed_input)
        processed_output = self.process_output(output)
        return processed_output

    def __str__(self):
        return f"""CLIP_Pipeline{{\nlabel_texts: {self.label_texts}\n{super().__str__()}}}"""


if __name__ == "__main__":
    p = CLIP_Pipeline("openai/clip-vit-base-patch32", ["test"], "cuda")
    print(p)