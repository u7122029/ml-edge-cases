from .base_pipeline import Model_Pipeline
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch
from tqdm import tqdm


class BLIP_Pipeline(Model_Pipeline):
    def __init__(self, weights_name, label_texts, device="cpu"):
        super().__init__("blip", weights_name, device)
        self.label_texts = label_texts

    def _Model_Pipeline__get_model(self):
        self.processor = BlipProcessor.from_pretrained(self.weights_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.weights_name)

    def _Model_Pipeline__put_on_device(self, device):
        super()._Model_Pipeline__put_on_device(device)
        self.model = self.model.to(self.device)

    def get_image_features(self, batch):
        outputs = []
        for image in tqdm(batch, total=len(batch)):
            input = self.processor(images=[image], text="x", return_tensors="pt").to("cuda")
            output = self.model(**input).last_hidden_state
            outputs.append(output.cpu().detach().flatten())
        outputs = torch.stack(outputs)
        return outputs

    def process_input(self, images):
        return [
            self.processor(text=self.label_texts, images=image, return_tensors="pt", padding=True).to(self.device)
            for image in images
        ]

    def forward(self, inputs):
        return torch.stack([self.model(**inp).itm_score[:,1].flatten() for inp in inputs])

    def process_output(self, outputs):
        return outputs.softmax(dim=1)

    def __call__(self, batch):
        processed_input = self.process_input(batch)
        output = self.forward(processed_input)
        processed_output = self.process_output(output)
        return processed_output

    def __str__(self):
        return f"""BLIP_Pipeline{{\nlabel_texts: {self.label_texts}\n{super().__str__()}}}"""