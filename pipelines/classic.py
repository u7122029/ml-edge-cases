from .base_pipeline import Model_Pipeline
import torch
import torchvision.models as tvm


class Classic_Pipeline(Model_Pipeline):
    def __init__(self, weights_name, dataset_name, device="cpu"):
        self.dataset_name = dataset_name
        super().__init__("classic", weights_name, device)

    def _Model_Pipeline__put_on_device(self, device):
        super()._Model_Pipeline__put_on_device(device)
        self.model = self.model.to(device)

    def get_image_features(self, batch):
        pass # TODO: Implement this eventually.

    def _Model_Pipeline__get_model(self):
        if self.dataset_name == "cifar10":
            self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{self.weights_name}",
                           pretrained=True, trust_repo=True)
        elif self.dataset_name == "imagenet":
            self.model = tvm.get_model(self.weights_name, weights="DEFAULT")

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

    def __str__(self):
        return f"""Clip_Pipeline{{\ndataset_name: {self.dataset_name}\n{super().__str__()}}}"""