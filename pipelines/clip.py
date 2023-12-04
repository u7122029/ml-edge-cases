from .base_pipeline import Model_Pipeline
from transformers import CLIPProcessor, CLIPModel


class CLIP_Pipeline(Model_Pipeline):
    def __init__(self, model_name, label_texts, device="cpu"):
        super().__init__(model_name, device)
        self.label_texts = label_texts

    def _Model_Pipeline__get_model(self):
        self.processor = CLIPProcessor.from_pretrained(f"openai/{self.model_name}")
        self.model = CLIPModel.from_pretrained(f"openai/{self.model_name}")

    def _Model_Pipeline__put_on_device(self, device):
        super()._Model_Pipeline__put_on_device(device)
        self.model = self.model.to(self.device)

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

    def __str__(self):
        return f"""CLIP_Pipeline{{\nlabel_texts: {self.label_texts}\n{super().__str__()}}}"""


if __name__ == "__main__":
    p = CLIP_Pipeline("clip-vit-base-patch32", ["test"], "cuda")
    print(p)