import base64
from io import BytesIO
import argparse

import umap
from PIL import Image
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from bokeh.palettes import Inferno10, Category10, TolRainbow, varying_alpha_palette, Viridis256, Plasma256, Turbo256
from bokeh.plotting import figure
from models import get_pipeline

from constants import *
from main import get_dataset

parser = argparse.ArgumentParser(description="Clusterer")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run.",
    choices=VALID_MODELS
)
parser.add_argument(
    "--image-noun",
    required=False,
    type=str,
    help="The image noun to use for clip models.",
    default=""
)
parser.add_argument(
    "--dataset",
    required=False,
    default="cifar10-test",
    help="The name of the dataset that should be used.",
    choices=["imagenet-val", "cifar10-test"]
)
parser.add_argument(
    "--prefix-mod",
    required=False,
    type=str,
    help="The prefix modifier for all ground_labels.",
    default=""
)
parser.add_argument(
    "--suffix-mod",
    required=False,
    type=str,
    help="The suffix modifier for all ground_labels.",
    default=""
)
parser.add_argument(
    "--data-root",
    required=False,
    default="C:/ml_datasets",
    type=str,
    help="path containing all datasets (training and validation)"
)
parser.add_argument(
    "--results-path",
    required=False,
    default="results",
    type=str,
    help="The path to store results."
)
parser.add_argument(
    "--figures-path",
    required=False,
    default=FIGURES_PATH_DEFAULT,
    type=str,
    help="The path to store figures."
)

class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.permute(1,2,0).flatten()


def embeddable_image1(data):
    img_data = data.reshape(3,32,32).permute(1,2,0)
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
    colours = Turbo256#Plasma256#Viridis256#varying_alpha_palette("#FF0000", 60,10)
    args = parser.parse_args()

    data_root = args.data_root
    dataset_full = args.dataset
    dataset_name, dataset_split = dataset_full.split("-")

    model_name = args.model
    model_type, weights_name = model_name_parser(model_name)

    results_path = args.results_path
    image_noun = args.image_noun
    prefix_mod = args.prefix_mod
    suffix_mod = args.suffix_mod
    figures_path = args.figures_path

    results_file_path = get_output_path(results_path, dataset_full, model_type, weights_name,
                                        image_noun, prefix_mod, suffix_mod)
    figures_file_path = get_output_path(figures_path, dataset_full, model_type, weights_name,
                                        image_noun, prefix_mod, suffix_mod, filetype="html")
    figures_file_path.parent.mkdir(parents=True,exist_ok=True)

    results = torch.load(str(results_file_path))
    labels = results["labels"]
    top3preds = results["top10preds"].to(torch.int16)
    top3confs = results["top10confs"]

    top1preds = top3preds[:,0]
    top1confs = top3confs[:,0]
    incorrect = torch.where(top1preds != labels)[0]

    dataset, _ = get_dataset("cifar10",
                             "test",
                             DATA_PATH_DEFAULT,
                             True,
                             transform=Compose([PILToTensor()]))

    pipeline = get_pipeline(model_type, weights_name, dataset_name).to(DEVICE)

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
    features = pipeline.get_image_features(incorrect_images)
    embedding = reducer.fit_transform(features)

    classes_df = pd.DataFrame(embedding, columns=["x","y"])
    #print(classes_df)
    classes_df["pred_pos"] = list(map(lambda x: str(x.item()), pred_poses))
    classes_df["label"] = [str(x.item()) for x in actual_labels]
    classes_df["label_text"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in actual_labels]
    classes_df["image"] = list(map(embeddable_image1, incorrect_images))
    classes_df["pred1"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 0]]
    classes_df["pred2"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 1]]
    classes_df["pred3"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 2]]

    classes_df["conf1"] = incorrect_confs[:, 0]
    classes_df["conf1"] = classes_df["conf1"].astype(str)
    classes_df["conf2"] = [str(x.item()) for x in incorrect_confs[:, 1]]
    classes_df["conf3"] = [str(x.item()) for x in incorrect_confs[:, 2]]

    top10_diff = 2 ** (-torch.Tensor(list(range(10))))
    top10_diff = top10_diff.repeat(len(incorrect_confs), 1)
    top10_diff *= incorrect_confs
    top10_diff = top10_diff[:, 0] - top10_diff[:, 1:].sum(dim=1)

    classes_df["top10_diff"] = top10_diff
    classes_df["top10_diff_str"] = classes_df["top10_diff"].astype(str)
    classes_df["top10_diff"] *= len(colours)
    classes_df["top10_diff"] = classes_df["top10_diff"].astype(int).astype(str)

    classes_df["conf1_idx"] = (classes_df["conf1"].astype(float) * len(colours)).astype(int).astype(str)


    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(classes_df)

    datasource = ColumnDataSource(classes_df)
    color_mapping = CategoricalColorMapper(factors=list(map(str, range(len(colours)))),
                                           palette=colours)

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
            <span style='font-size: 18px'>@label_text</span>
            <br>
            <span style='font-size: 16px; color: #224499'>Top10_diff:</span>
            <span style='font-size: 18px'>@top10_diff_str</span>
            <br>
            <table>
                <tr>
                    <td><strong>@pred1</strong></td>
                    <td><strong>@conf1</strong></td>
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
        color=dict(field='top10_diff', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    output_file(str(figures_file_path), mode='inline')
    show(plot_figure)