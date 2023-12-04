import pandas as pd
import torch
import umap
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from bokeh.palettes import Spectral10, Spectral3, Inferno3, Colorblind3
from bokeh.plotting import figure

from main import get_dataset
from constants import *


class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.permute(1,2,0).flatten()


def embeddable_image1(data):
    img_data = data.reshape(32,32,3)
    image = Image.fromarray(img_data.numpy(), mode='RGB').resize((64, 64), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


def get_pred_positions(top_preds, ground_labels, unique_labels):
    output = torch.zeros(top_preds.shape[0]).int() # 0 means not in top k.
    for label in unique_labels:
        indices = torch.where(ground_labels == label)[0]
        label_preds = top_preds[indices]
        detections = torch.where(label_preds == label)
        output[indices[detections[0].int()]] = detections[1].int()
    return output


if __name__ == "__main__":
    # Load results file
    results = torch.load("results/cifar10-test/clip-vit-large-patch14-336.pt")
    labels = results["labels"]
    top3preds = results["top3preds"].to(torch.int16)
    top3confs = results["top3confs"]

    top1preds = top3preds[:,0]
    top1confs = top3confs[:,0]
    incorrect = torch.where(top1preds != labels)[0]

    dataset, _ = get_dataset("cifar10",
                             "test",
                             DATA_PATH_DEFAULT,
                             True,
                             transform=Compose([PILToTensor(), Flatten()]))

    images = []
    labels = []
    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    incorrect_preds = top3preds[incorrect]
    incorrect_confs = top3confs[incorrect]
    incorrect_images = images[incorrect]
    actual_labels = labels[incorrect]
    pred_poses = get_pred_positions(incorrect_preds, actual_labels, list(range(10)))

    #torch.set_printoptions(profile="full")

    reducer = umap.UMAP(metric="cosine")
    embedding = reducer.fit_transform(incorrect_images)

    classes_df = pd.DataFrame(embedding, columns=["x","y"])
    #print(classes_df)
    classes_df["pred_pos"] = list(map(lambda x: str(x.item()), pred_poses))
    classes_df["label"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in actual_labels]
    classes_df["image"] = list(map(embeddable_image1, incorrect_images))
    classes_df["pred1"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 0]]
    classes_df["pred2"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 1]]
    classes_df["pred3"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 2]]
    classes_df["conf1"] = [str(x.item()) for x in incorrect_confs[:, 0]]
    classes_df["conf2"] = [str(x.item()) for x in incorrect_confs[:, 1]]
    classes_df["conf3"] = [str(x.item()) for x in incorrect_confs[:, 2]]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(classes_df)

    datasource = ColumnDataSource(classes_df)
    color_mapping = CategoricalColorMapper(factors=[str(2 - x) for x in range(3)],
                                           palette=Colorblind3)

    plot_figure = figure(
        title='UMAP projection of the CIFAR10 dataset',
        # plot_width=600,
        # plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Pred_pos:</span>
            <span style='font-size: 18px'>@pred_pos</span>
            <br>
            <span style='font-size: 16px; color: #224499'>Label:</span>
            <span style='font-size: 18px'>@label</span>
            <br>
            <table>
                <tr>
                    <td>@pred1</td>
                    <td>@conf1</td>
                </tr>
                <tr>
                    <td>@pred2</td>
                    <td>@conf2</td>
                </tr>
                <tr>
                    <td>@pred3</td>
                    <td>@conf3</td>
                </tr>
            </table>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='pred_pos', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)