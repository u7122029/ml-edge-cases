from abc import ABC, abstractmethod

class Model_Pipeline(ABC):
    def __init__(self, model_type, weights_name, device):
        self.device = device
        self.model_type = model_type
        self.weights_name = weights_name

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
    def __call__(self, batch):
        """
        Runs a batch of input items through the pipeline.
        :param batch: The batch of inputs
        :return: The batch of outputs to return (eg: confidence scores, bounding boxes, etc)
        """

        raise NotImplementedError("Method has not yet been implemented.")

    @abstractmethod
    def __str__(self):
        return f"""device: {self.device}\nmodel_name: {self.model_name()}"""