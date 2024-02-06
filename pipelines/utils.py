import torch


def PIL_collate_fn(x):
    batch = [a[0] for a in x]
    labels = torch.Tensor([a[1] for a in x])
    return batch, labels