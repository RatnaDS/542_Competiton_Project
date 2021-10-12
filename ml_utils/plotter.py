import os
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(train_loss, val_loss, train_acc, val_acc, 
                x=None, x_label="Epoch", show=True, save=False, save_path=None):
    
    if save: assert save_path is not None

    if x is None:
        x = np.arange(len(train_loss))

    f1 = plt.figure(dpi=200)

    tloss_plot, = plt.plot(x, train_loss, c="r")
    vloss_plot, = plt.plot(x, val_loss, c="b")
    plt.legend((tloss_plot, vloss_plot), ("Train Loss", "Validation Loss"))
    plt.xlabel(x_label)
    plt.title("Loss")

    if save:
        plt.savefig(os.path.join(save_path, "loss.png"), bbox_inches="tight", pad_inches=0.1)

    f2 = plt.figure(dpi=200)
    tacc_plot, = plt.plot(x, train_acc, c="r")
    vacc_plot, = plt.plot(x, val_acc, c="b")
    plt.legend((tacc_plot, vacc_plot), ("Train Accuracy", "Validation Accuracy"))
    plt.xlabel(x_label)
    plt.title("Accuracy")

    if save:
        plt.savefig(os.path.join(save_path, "acc.png"), bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()
