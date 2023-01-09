from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def set_default(figsize=(10, 10), dpi=100):
    # plt.style.use(['dark_background', 'bmh'])
    # plt.rc('axes', facecolor='k')
    # plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10")
    plt.title("Moons Dataset - Noise: $0.1$")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()

def plot_loss(title, losses, scores):
    plt.rcParams["axes.grid"] = True
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
    fig.suptitle(title)
    ax1.plot(np.linspace(1, len(losses), len(losses)), losses)
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(0, len(losses))

    ax1.plot(np.linspace(1, len(scores), len(scores)), scores)
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(0, len(losses))

    ax1.legend(["Acc", "Loss"])

    plt.show()

def plot_confusion_matrix(y, y_pred):
    # Confusion matrix
    conf_mat = confusion_matrix(y, y_pred.detach().numpy())

    ax = sns.heatmap(conf_mat, annot=True, fmt="g", linewidths=0.5, cmap=plt.cm.Blues)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    # ax.figure.savefig('Dataset c3 - Model 48 - 1 - 3.png', dpi=300)
    plt.show()

    print(classification_report(y, y_pred.detach().numpy()))