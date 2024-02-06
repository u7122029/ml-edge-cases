import torch
from PIL import Image
from base_pipeline import Model_Pipeline
from tqdm import tqdm
from torchvision.transforms import PILToTensor

from lavis.models import load_model_and_preprocess

class BLIP_LAVIS_Pipeline(Model_Pipeline):
    def __init__(self, weights_name, label_texts, device="cpu"):
        super().__init__("blip", "base", device)
        self.label_texts = label_texts

    def _Model_Pipeline__get_model(self):
        self.model, self.processor, _ = load_model_and_preprocess("blip_feature_extractor",
                                                                  model_type=self.weights_name,
                                                                  is_eval=True,
                                                                  device=self.device)

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
        out = torch.stack([self.model(**inp).itm_score for inp in inputs]).softmax(dim=2)[:,:,1]
        return out

    def process_output(self, outputs):
        return outputs.softmax(dim=1)

    def __call__(self, batch):
        processed_input = self.process_input(batch)
        output = self.forward(processed_input)
        processed_output = self.process_output(output)
        return processed_output

    def __str__(self):
        return f"""BLIP_Pipeline{{\nlabel_texts: {self.label_texts}\n{super().__str__()}}}"""

raw_image = Image.open("merlion.png").convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess("blip_feature_extractor", model_type="base", is_eval=True, device=device)

#cls_names = ["merlion", "sky", "giraffe", "fountain", "marina bay"]
cls_names = ["boat", "fountain", "river"]

# (optional) add prompt when we want to use the model for zero-shot classification
from lavis.processors.blip_processors import BlipCaptionProcessor

cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

sample = {"image": image, "text_input": cls_names}

image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
text_features = model.extract_features(sample, mode="text").text_embeds_proj[:, 0]

sims = (image_features @ text_features.t())[0] / model.temp
probs = torch.nn.Softmax(dim=0)(sims).tolist()

for cls_nm, prob in zip(cls_names, probs):
    print(f"{cls_nm}: \t {prob:.3%}")