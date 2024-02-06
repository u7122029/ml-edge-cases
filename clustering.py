import base64
from io import BytesIO
import argparse

import umap
from PIL import Image
from bokeh.io import show, output_file
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CategoricalColorMapper, HoverTool
from bokeh.palettes import interp_palette
from bokeh.plotting import figure
from models import get_pipeline

from constants import *
from main import get_dataset
import torch
from sklearn.linear_model import LinearRegression
import pacmap
from torch.utils.data import Subset
from tqdm import tqdm

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
    default="photo"
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
parser.add_argument(
    "--use-random-confs",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to use random prediction confidences - mainly to test the validity of the scatter plot on the right."
)


class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.permute(1,2,0).flatten()


def embeddable_image1(data):
    #img_data = data.reshape(3,32,32).permute(1,2,0)
    #image = Image.fromarray(img_data.numpy(), mode='RGB').resize((64, 64), Image.Resampling.BICUBIC)
    image = data.resize((64, 64), Image.Resampling.BICUBIC)
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


def generate_topk_diff(incorrect_confs, k=10, n_weight=2):
    """
    Generates the topk diff measure
    :param top_confs: N * M tensor where N is the number of probabilitity vectors to consider, '
                        and M is the length of each.
    :param k: The top-k of confidence scores to consider. Must be less than or equal to M. Default is 10.
                If M <= 10 then take k = M.
    :return: Tensor of size N, with each entry corresponding to the topk weighted differences.
    """
    r = 1/n_weight
    k = max(incorrect_confs.shape[1], k)
    top10_diff = r ** (torch.Tensor(list(range(k))))
    top10_diff = top10_diff.repeat(len(incorrect_confs), 1)
    top10_diff *= incorrect_confs
    top10_diff = top10_diff[:, 0] - top10_diff[:, 1:].sum(dim=1)
    return top10_diff


def avg_knn_dist(points, weighted_confs, k=10):
    # points is a N by M tensor
    k = min(k + 1, len(points))
    dist_mat = torch.norm(points[:, None, :] - points[None, :, :], dim=2)#cosine_similarity(points[:, None, :], points[None, :, :], dim=-1, eps=1e-11)
    dist_mat = dist_mat / dist_mat.max() # Shorter distance gives bigger weight.
    tops = torch.topk(dist_mat,k=k,dim=1, largest=False)
    top_dists = tops.values[:, 1:]
    top_idxs = tops.indices[:, 1:]
    #top_confs = weighted_confs[top_idxs]

    return torch.mean(top_dists, dim=1)


def weighted_avg_knn_dist(points, weighted_confs, k=None):
    # points is a N by M tensor
    if not k: k = len(points)
    else: k = min(k + 1, len(points))
    weighted_conf_diffs = 1 - torch.abs(weighted_confs[:,None] - weighted_confs)
    dist_mat = torch.norm(points[:, None, :] - points[None, :, :], dim=2)#cosine_similarity(points[:, None, :], points[None, :, :], dim=-1, eps=1e-11)
    dist_mat = dist_mat / dist_mat.max() # Shorter distance gives bigger weight.

    tops = torch.topk(dist_mat, k=k, dim=1, largest=False)
    top_dists = tops.values[:, 1:]
    top_idxs = tops.indices[:, 1:]

    selected_diffs = []
    for row_idx in range(len(weighted_conf_diffs)):
        current_weighted_conf_diff = weighted_conf_diffs[row_idx]
        current_top_idxs = top_idxs[row_idx]
        q = current_weighted_conf_diff[current_top_idxs]
        selected_diffs.append(q)

    selected_diffs = torch.stack(selected_diffs)

    #top_confs = weighted_confs[top_idxs]

    return torch.mean(top_dists * selected_diffs, dim=1)


def construct_random_data(N = 2000):
    confs = []
    for i in range(N):
        a = []
        accum = 1
        rands = torch.rand(10)
        for j in range(9):
            inp = rands[j].item() * accum
            a.append(inp)
            accum -= inp
        a.append(accum)
        confs.append(a)
    confs = torch.sort(torch.Tensor(confs), dim=1,descending=True).values

    labels = torch.zeros(N).to(torch.int8)
    top10preds = torch.ones(N, 10).int().to(torch.int16)

    return labels, top10preds, confs


def generate_plots(classes_df, figures_file_path, colours):
    datasource = ColumnDataSource(classes_df)
    color_mapping = CategoricalColorMapper(factors=list(map(str, range(len(colours)))),
                                           palette=colours)

    scatter_figure = figure(
        title='UMAP projection of the CIFAR10 dataset',
        # plot_width=600,
        # plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    scatter_figure.add_tools(HoverTool(tooltips="""
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

    scatter_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='conf1_idx', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )

    scatter_figure1 = figure(
        title='Average KNN Dist vs. Confidence Difference',
        # plot_width=600,
        # plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )
    scatter_figure1.circle(
        'conf1_raw',
        'k_nearest_avgs',
        source=datasource,
        # color=dict(field='top10_diff', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    # lr = LinearRegression
    output_file(str(figures_file_path), mode='inline')
    show(row(scatter_figure, scatter_figure1))


def perform_umap(pipeline, images):
    reducer = umap.UMAP(metric="cosine")
    features = pipeline.get_image_features(images)
    embedding = reducer.fit_transform(features)
    return embedding


def perform_pacmap(image_features):
    reducer = pacmap.PaCMAP()
    embedding = reducer.fit_transform(image_features)
    return embedding


def main(args):
    # Load results file
    colours = interp_palette(("#0000FF", "#FFFF00"), 256)

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
    use_random_confs = args.use_random_confs

    results_file_path = get_output_path(results_path, dataset_full, model_type, weights_name,
                                            image_noun, prefix_mod, suffix_mod)
    figures_file_path = get_output_path(figures_path, dataset_full, model_type, weights_name,
                                            image_noun, prefix_mod, suffix_mod, filetype="html")
    figures_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not use_random_confs:
        results = torch.load(str(results_file_path))
        labels = results["labels"]
        top10preds = results["top10preds"].to(torch.int16)
        top10confs = results["top10confs"]
    else:
        labels, top10preds, top10confs = construct_random_data(4000)

    top1preds = top10preds[:, 0]
    #top1confs = top10confs[:, 0]
    incorrect = torch.where(top1preds != labels)[0]

    dataset_conf, _ = get_dataset("cifar10",
                             "test",
                             DATA_PATH_DEFAULT,
                             indices=incorrect)
    dataset = dataset_conf()

    pipeline = get_pipeline(model_type, weights_name, dataset_name).to(DEVICE)

    incorrect_preds = top10preds[incorrect]
    incorrect_confs = top10confs[incorrect]

    incorrect_images = []
    actual_labels = []
    for image, label in dataset:
        incorrect_images.append(image)
        actual_labels.append(label)

    actual_labels = torch.tensor(actual_labels)

    pred_poses = get_pred_positions(incorrect_preds, actual_labels, list(range(10)))

    # torch.set_printoptions(profile="full")
    image_features = pipeline.get_image_features(dataset_conf)

    if not use_random_confs:
        embedding = perform_pacmap(image_features)
    else:
        embedding = torch.rand(4000,2)

    classes_df = pd.DataFrame(embedding, columns=["x", "y"])
    # classes_df["dist_from_centre"]
    classes_df["pred_pos"] = pred_poses.tolist()
    classes_df["pred_pos"] = classes_df["pred_pos"].astype(str)

    classes_df["label"] = actual_labels
    classes_df["label"] = classes_df["label"].astype(str)

    classes_df["label_text"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in actual_labels]
    classes_df["image"] = list(map(embeddable_image1, incorrect_images))
    classes_df["pred1"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 0]]
    classes_df["pred2"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 1]]
    classes_df["pred3"] = [str(CIFAR10_LABELS_TEXT[x.item()]) for x in incorrect_preds[:, 2]]

    classes_df["conf1_raw"] = incorrect_confs[:, 0].tolist()
    classes_df["conf1"] = classes_df["conf1_raw"].astype(str)
    classes_df["conf2"] = [str(x.item()) for x in incorrect_confs[:, 1]]
    classes_df["conf3"] = [str(x.item()) for x in incorrect_confs[:, 2]]

    top10_diff = generate_topk_diff(incorrect_confs)

    classes_df["top10_diff_original"] = top10_diff
    classes_df["top10_diff_str"] = classes_df["top10_diff_original"].astype(str)
    classes_df["top10_diff"] = classes_df["top10_diff_original"] * len(colours)
    classes_df["top10_diff"] = classes_df["top10_diff"].astype(int).astype(str)

    classes_df["conf1_idx"] = (classes_df["conf1_raw"] * len(colours)).astype(int).astype(str)
    classes_df["k_nearest_avgs"] = avg_knn_dist(image_features, top10_diff, k=20)

    generate_plots(classes_df,figures_file_path,colours)


if __name__ == "__main__":
    #labels, preds, confs = construct_random_data()
    args = parser.parse_args()
    main(args)
