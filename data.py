from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets import CIFAR10
from typing import Type, Sequence, Union


class DatasetConfig:
    """
    Class which allows the dynamic editing of pytorch dataset parameters.
    """
    def __init__(self,
                 dataset: Type[VisionDataset],
                 indices: Union[Sequence[int], None] = None,
                 **dataset_kwargs):
        self._dataset = dataset
        self._dataset_kwargs = dataset_kwargs

        self.indices: Union[Sequence[int], None] = indices

    def __contains__(self, item):
        return item in self._dataset_kwargs

    def __getitem__(self, key):
        return self._dataset_kwargs[key]

    def __setitem__(self, key, value):
        self._dataset_kwargs[key] = value

    def update(self, d: dict):
        self._dataset_kwargs.update(d)

    def __call__(self):
        dataset = self._dataset(**self._dataset_kwargs)

        if self.indices is not None: dataset = Subset(dataset, self.indices)
        return dataset


if __name__ == "__main__":
    x = DatasetConfig(CIFAR10, **dict(root="C:/ml_datasets", train=False))
    print(x())
