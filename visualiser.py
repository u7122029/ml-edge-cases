import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, VPacker, TextArea
from torchvision.transforms import PILToTensor
from main import get_dataset
from constants import *
import numpy as np; np.random.seed(42)
import argparse

parser = argparse.ArgumentParser(description="ModelOutput Visualiser")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="The classifier to run." # TODO: ADD VALID CHOICES
)
parser.add_argument("--use-clip",
                    action=argparse.BooleanOptionalAction)
parser.add_argument(
    "--image-noun",
    required=False,
    type=str,
    help="The image noun to use for clip models.",
    default=None
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
    "--dataset",
    required=False,
    default="cifar10-test",
    help="The name of the dataset that should be used.",
    choices=["imagenet-val", "cifar10-test"]
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

def load_results(dset_name, modelname):
    return torch.load(f"results/{dset_name}/{modelname}.pt")


def generate_plot(dset, indices, top3preds, top3confs, labels_text, zoom=0.5):
    print("gp")
    #selected_images = np.array([img for img, _ in dset])[indices]
    #selected_labels = np.array([label for _,label in dset])[indices]
    selected_preds = top3preds[indices]
    selected_confs = top3confs[indices]

    x = selected_confs[:,0]
    y = np.zeros(len(x))

    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line = ax.scatter(x,y, ls="", marker=".")

    # create the annotations box
    im = OffsetImage(np.zeros((32,32,3)), zoom=zoom)
    text = TextArea("ABCDEF")
    content = VPacker(children=[im, text], sep=3)

    xybox=(120., 120.) # offset
    ab = AnnotationBbox(content, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))

    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            inds = line.contains(event)[1]["ind"]
            choice = inds[torch.argmin(torch.abs(x[inds] - event.x)).item()]

            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)

            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            ab.set_visible(True)
            ab.xy =(x[choice], y[choice])
            im.set_data(dset[indices[choice]][0].permute(1,2,0))
            text.set_text("Predicted:\n" +
                          "\n".join([f"{labels_text[label.item()]}, {round(conf.item(),4)}" for label, conf in zip(selected_preds[choice], selected_confs[choice])]) +
                          "\n" +
                          f"Actual:\n{labels_text[dset[indices[choice]][1]]}"
                          )
        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)


def generate_class_hist(top1preds, dataset):
    plt.figure()
    if dataset == "cifar10":
        bins = 10
    else:
        bins = 1000
    plt.hist(top1preds, bins=bins)



if __name__ == "__main__":
    print(DEVICE)
    args = parser.parse_args()
    model_name = args.model
    image_noun = args.image_noun
    dataset_name, split = args.dataset.split("-")
    data_root = args.data_root
    use_clip = args.use_clip
    prefix_mod = args.prefix_mod
    suffix_mod = args.suffix_mod
    results_path = args.results_path

    dset, labels_text = get_dataset(dataset_name, split, data_root, use_clip, visualise=True)
    results = load_results(f"{dataset_name}-{split}", model_name)
    labels = results["ground_labels"]
    top3preds = results["top3preds"].to(torch.int16)
    top3confs = results["top3confs"]
    top1preds = top3preds[:,0]
    incorrect = torch.where(top1preds != labels)[0]
    print(incorrect.shape)
    print(len(top3confs))
    if dataset_name == "cifar10":
        zoom = 2
    else:
        zoom = 0.5
    generate_plot(dset, incorrect, top3preds, top3confs, labels_text, zoom)
    generate_class_hist(top1preds, dataset_name)
    plt.show()