from abc import ABC, abstractmethod


class Base_Pipeline(ABC):
    def __init__(self, pre_image_transform=None, pre_label_transform=None, device="cpu"):
        self.pre_image_transform = pre_image_transform
        self.pre_label_transform = pre_label_transform
        self.device = device

    @abstractmethod
    def to(self, device):
        self.device = device

    def get_image_features(self, dataset_func, label_texts, **kwargs):
        """
        Runs the pipeline over the dataset function.
        :param dataset_func: function (pre_image_transform, pre_label_transform) -> dataset
        :return: the image features for each image in the dataset.
        """
        pass

    @abstractmethod
    def __call__(self, dataset_func, label_texts, **kwargs):
        """
        Runs the pipeline over the dataset function.
        :param dataset_func: function (pre_image_transform, pre_label_transform) -> dataset
        :return: desired output to return after applying model to dataset.
        """
        pass


class Model_Pipeline(ABC):
    def __init__(self, model_type, weights_name, dataset_transform, device="cpu"):
        """
        Initialisor for a Model Pipeline.
        :param model_type: The type of the model, eg: clip, blip, etc.
        :param weights_name: The model weights to use.
        :param dataset_transform: The torchvision transform to apply over each image in the dataset.
        :param device: The device eg: "cpu", "cuda", etc.
        """
        self.device = device
        self.model_type = model_type
        self.weights_name = weights_name
        self.dataset_transform = dataset_transform

        self.__get_model()
        self.__put_on_device(self.device)

    @abstractmethod
    def __get_model(self):
        pass

    @abstractmethod
    def __put_on_device(self, device):
        self.device = device

    def to(self, device):
        """
        Puts the model as well as other necessary parts of the pipeline to the specified device.
        :param device: The device to put the pipeline on.
        :return: The model itself.
        """
        self.__put_on_device(device)
        return self

    def model_name(self):
        return f"{self.model_type}_{self.weights_name}"

    def get_model(self):
        return str(self.model)

    @abstractmethod
    def get_image_features(self, batch):
        """
        Takes a batch of images and extracts features from each image.
        :param batch: PyTorch Tensor of size B * H * W * NUM_CHANNELS
        :return: PyTorch Tensor of size B * EMBED_SIZE
        """
        pass

    @abstractmethod
    def __call__(self, dataset_func):
        """
        Runs a dataset through the pipeline.
        :param dataset_func: A function transform -> pytorch dataset which applies the transform to the dataset.
        :return: The batch of outputs to return (eg: confidence scores, bounding boxes, etc)
        """

        raise NotImplementedError("Method has not yet been implemented.")

    @abstractmethod
    def __str__(self):
        return f"""device: {self.device}\nmodel_name: {self.model_name()}"""