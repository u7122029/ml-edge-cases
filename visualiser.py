import matplotlib.pyplot as plt
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, VPacker, TextArea
from main import get_dataset, ToNumpyArray
from constants import *
import numpy as np; np.random.seed(42)


def load_results(dset_name, modelname):
    return torch.load(f"results/{dset_name}/{modelname}.pt")


def generate_plot(dset, indices, top3preds, top3confs):
    selected_images = np.array([img for img, _ in dset])[indices]
    selected_labels = np.array([label for _,label in dset])[indices]
    selected_preds = top3preds[indices]
    selected_confs = top3confs[indices]

    x = selected_confs[:,0]
    y = np.zeros(len(x))

    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line = ax.scatter(x,y, ls="", marker=".")

    # create the annotations box
    im = OffsetImage(selected_images[0,:,:], zoom=2)
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
            im.set_data(selected_images[choice,:,:])
            text.set_text("Predicted:\n" +
                          "\n".join([f"{CIFAR10_LABELS_TEXT[label.item()]}, {round(conf.item(),4)}" for label, conf in zip(selected_preds[choice], selected_confs[choice])]) +
                          "\n" +
                          f"Actual:\n{CIFAR10_LABELS_TEXT[selected_labels[choice]]}"
                          )
        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)


if __name__ == "__main__":
    dset = get_dataset("CIFAR10", root="C:/ml_datasets", transform=ToNumpyArray(), train=False)
    results = load_results("cifar10-test", "clip")
    labels = torch.Tensor([label for _, label in dset])
    top3preds = results["top3preds"].to(torch.uint8)
    top3confs = results["top3confs"]
    incorrect = torch.where(top3preds[:,0] != labels)[0]
    generate_plot(dset, incorrect, top3preds, top3confs)
    plt.show()