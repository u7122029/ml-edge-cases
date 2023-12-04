import pandas as pd
import torch
import umap
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from bokeh.io import show
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from bokeh.palettes import Spectral10
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


if __name__ == "__main__":
    # Load results file
    results = torch.load("results/cifar10-test/clip-vit-base-patch32_photo.pt")
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

    incorrect_images = images[incorrect]
    actual_labels = labels[incorrect]

    reducer = umap.UMAP(metric="cosine")
    embedding = reducer.fit_transform(incorrect_images)

    classes_df = pd.DataFrame(embedding, columns=["x","y"])
    #print(classes_df)
    classes_df["label"] = [str(x.item()) for x in actual_labels]
    classes_df["image"] = list(map(embeddable_image1, incorrect_images))

    datasource = ColumnDataSource(classes_df)
    color_mapping = CategoricalColorMapper(factors=[str(10 - x) for x in range(10)],
                                           palette=Spectral10)

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
            <span style='font-size: 16px; color: #224499'>Digit:</span>
            <span style='font-size: 18px'>@digit</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='label', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)