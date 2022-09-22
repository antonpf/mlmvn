# %% [markdown]
# # Sensorless Drive Diagnosis
# 
# > In this example, the main focus is the classification of individual states of a motor.

# %%
# |hide
from nbdev.showdoc import *

# %%
# | hide
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import copy

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from matplotlib import pyplot as plt
import seaborn as sns

from mlmvn.layers import FirstLayer, HiddenLayer, OutputLayer, cmplx_phase_activation
from mlmvn.loss import ComplexMSELoss, ComplexMSE_adjusted_error
from mlmvn.optim import MySGD, ECL
from pathlib import Path
from clearml import Task, Logger

# %%
# | hide
# --- helper functions ---
def reverse_one_hot(x, neuronCats):
    a = np.zeros(len(x))
    x = torch.detach(x)
    for i in range(len(x)):
        a[i] = torch.max(x[i]) - 1 + np.argmax(x[i]) * neuronCats
    return a


def accuracy(out, yb):
    out = out.type(torch.double)
    yb = yb.type(torch.double)
    x = 0
    for i in range(len(out)):
        x += torch.equal(out[i], yb[i])
    return x / len(out)


def prepare_data(x_train, x_valid, y_train, y_valid, neuronCats):
    # one-hot encoding
    numSamples, numFeatures = x_valid.shape
    y_valid_int = y_valid
    y2 = y_valid + 1  # auxiliary variable so that classes start at 1 and not 0
    numClasses = max(y2)
    target_ids = range(numClasses)
    no = int(np.ceil(numClasses / neuronCats))  # number of output neurons
    if no != 1:
        y_valid = torch.zeros(numSamples, no)
        for i in range(numSamples):
            k = int(np.ceil(y2[i] / neuronCats)) - 1
            c = np.mod((y2[i] - 1), neuronCats) + 1
            y_valid[i, k] = c
    numSamples, numFeatures = x_train.shape
    y_train_int = y_train
    y2 = y_train + 1  # auxiliary variable so that classes start at 1 and not 0
    if no != 1:
        y_train = torch.zeros(numSamples, no)
        for i in range(numSamples):
            k = int(np.ceil(y2[i] / neuronCats)) - 1
            c = np.mod((y2[i] - 1), neuronCats) + 1
            y_train[i, k] = c
    del y2

    # Convert numpy arrays into torch tensors
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    if y_train.size().__len__() == 1:
        y_train = torch.unsqueeze(y_train, 1)
        y_valid = torch.unsqueeze(y_valid, 1)

    # convert angles to complex numbers on unit-circle
    x_train = torch.exp(1.0j * x_train)
    x_valid = torch.exp(1.0j * x_valid)

    return x_train, x_valid, y_train, y_valid


def get_splitted_data(X, y, neuronCats):
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, y, train_size=46806, random_state=42
    )
    x_train, x_valid, y_train, y_valid = prepare_data(
        x_train, x_valid, y_train, y_valid, neuronCats
    )

    return x_train, x_valid, y_train, y_valid


def get_splitted_data_by_index(X, y, neuronCats, train_index, test_index):
    x_train, x_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    x_train, x_valid, y_train, y_valid = prepare_data(
        x_train, x_valid, y_train, y_valid, neuronCats
    )
    return x_train, x_valid, y_train, y_valid


# --- Plots ---
def plot_loss(title, losses, scores):
    plt.rcParams["axes.grid"] = True
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
    fig.suptitle("CVNN - Moons")
    ax1.plot(np.linspace(1, len(losses), len(losses)), losses)
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(0, len(losses))

    ax1.plot(np.linspace(1, len(scores), len(scores)), scores)
    ax1.set_xlabel("Epoch")
    ax1.set_xlim(0, len(losses))

    ax1.legend(["Acc", "Loss"])

    plt.show()


def plot_weights(title, ylabel_1, ylabel_2, weights_real, weights_imag):
    # y_min = np.min([np.min(weights_real), np.min(weights_imag)])
    # y_max = np.max([np.max(weights_real), np.max(weights_imag)])

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(14, 3))
    fig.suptitle(title)
    ax[0].plot(np.linspace(1, len(weights_real), len(weights_real)), weights_real)
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel(ylabel_1)
    # ax[0].set_title("Real Valued Weigts")
    ax[0].set_xlim(0, len(weights_real))
    # ax[0].set_ylim(y_min, y_max)

    ax[1].plot(np.linspace(1, len(weights_imag), len(weights_imag)), weights_imag)
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel(ylabel_2)
    # ax[1].set_title("Imaginary Valued Weights")
    ax[1].set_xlim(0, len(weights_imag))
    # ax[1].set_ylim(y_min, y_max)

    plt.show()


def plot_loss_acc_list(title, list_losses, list_scores, image_name):
    losses = np.mean(list_losses, axis=0)
    scores = np.mean(list_scores, axis=0)

    losses_std = np.std(list_losses, axis=0)
    scores_std = np.std(list_scores, axis=0)

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
    fig.suptitle(title)
    ax1.plot(np.linspace(1, len(losses), len(losses)), losses)
    ax1.fill_between(
        np.linspace(1, len(losses), len(losses)),
        losses + losses_std,
        losses - losses_std,
        alpha=0.5,
        linewidth=0,
    )

    ax1.plot(np.linspace(1, len(scores), len(scores)), scores)
    ax1.fill_between(
        np.linspace(1, len(scores), len(scores)),
        scores + scores_std,
        scores - scores_std,
        alpha=0.5,
        linewidth=0,
    )
    ax1.set_xlabel("Epoch")

    plt.legend(["Loss Mean", "Loss Std", "Acc. Mean", "Acc. Std"])
    fig.savefig(image_name, format="png", dpi=600)

    plt.show()
    # save
    # fig.savefig(image_name + ".svg", format="svg", dpi=600)


# --- Logging ---
model_dict: dict = {}


def fc_hook(layer_name, module, grad_input, grad_output):
    if layer_name in model_dict:
        model_dict[layer_name]["weights"] = module.weights.detach().clone()
        model_dict[layer_name]["bias"] = module.bias.detach().clone()
        model_dict[layer_name]["grad_input"] = grad_input
        model_dict[layer_name]["grad_output"] = grad_output
    else:
        model_dict[layer_name] = {}
        model_dict[layer_name]["weights"] = module.weights.detach().clone()
        model_dict[layer_name]["bias"] = module.bias.detach().clone()
        model_dict[layer_name]["grad_input"] = grad_input
        model_dict[layer_name]["grad_output"] = grad_output

# %%
# | hide
# control variables
# number of categories a neuron can distinguish / parameter that determines the number of output neurons
neuronCats = 1
# number of categories per neuron, i.e. neuronCats (+ 1 for others in case of multiple Outputs)
categories = 2
# how often a classification sector occurs (1 means no periodicity)
periodicity = 1
# path to store best model parameters

# %% [markdown]
# ## Load Data

# %%
train_csv = pd.read_csv(
    "/home/antonpfeifer/Documents/mlmvn/nbs/examples/autass/data/autass_data2.csv",
    header=None,
    dtype=np.double,
)
data = np.array(train_csv.values[:, 1:50])
del train_csv

# %%
X = data[:, 0:48]
y = data[:, 48].astype(int) - 1

yt = copy.copy(y)
yt[yt == 0] = 20
yt[yt == 1] = 21
yt[yt == 2] = 22
yt[yt == 3] = 23
yt[yt == 4] = 26
yt[yt == 5] = 24
yt[yt == 6] = 27
yt[yt == 7] = 29
yt[yt == 8] = 30
yt[yt == 9] = 25
yt[yt == 10] = 28
yt -= 20
y = yt
del yt

# %% [markdown]
# ## Config

# %%
epochs = 200
batch_size = 538
lr = 1

# %% [markdown]
# ## Single Layer

# %% [markdown]
# ### MLMVN [48-100-11]

# %%
PATH = str(Path.cwd() / "models/autass-mlmvn_48-100-11.pt")

# %%
class Model(nn.Module):
    def __init__(self, categories, periodicity):
        super().__init__()
        self.categories = categories
        self.periodicity = periodicity
        self.first_linear = FirstLayer(48, 100)
        self.phase_act1 = cmplx_phase_activation()
        self.linear_out = OutputLayer(100, 11)
        self.phase_act2 = cmplx_phase_activation()
        # Hooks
        self.first_layer_hook_handle = self.first_linear.register_full_backward_hook(
            self.first_layer_backward_hook
        )
        self.output_hook_handle = self.linear_out.register_full_backward_hook(
            self.output_layer_backward_hook
        )

    def forward(self, x):
        x = self.first_linear(x)
        x = self.phase_act1(x)
        x = self.linear_out(x)
        x = self.phase_act2(x)
        return x

    def first_layer_backward_hook(self, module, grad_input, grad_output):
        fc_hook("first_layer", module, grad_input, grad_output)

    def hidden_layer_backward_hook(self, module, grad_input, grad_output):
        fc_hook("hidden_layer", module, grad_input, grad_output)

    def output_layer_backward_hook(self, module, grad_input, grad_output):
        fc_hook("output_layer", module, grad_input, grad_output)

    def angle2class(self, x: torch.tensor) -> torch.tensor:
        tmp = x.angle() + 2 * np.pi
        angle = torch.remainder(tmp, 2 * np.pi)

        # This will be the discrete output (the number of sector)
        o = torch.floor(self.categories * self.periodicity * angle / (2 * np.pi))
        return torch.remainder(o, self.categories)

    def predict(self, x):
        """
        Performs the prediction task of the network

        Args:
          x: torch.Tensor
            Input tensor of size ([3])

        Returns:
          Most likely class i.e., Label with the highest score
        """
        # Pass the data through the networks
        output = self.forward(x)

        # # Choose the label with the highest score
        # return torch.argmax(output, 1)
        return self.angle2class(output)


def fit(model, X, y, epochs, batch_size, optimizer, criterion, categories, periodicity):
    # List of losses for visualization
    losses = []
    scores = []
    acc_best = 0

    for i in range(epochs):
        # Pass the data through the network and compute the loss
        # We'll use the whole dataset during the training instead of using batches
        # in to order to keep the code simple for now.

        batch_loss = []

        for j in range((X.shape[0] - 1) // batch_size + 1):
            start_j = j * batch_size
            end_j = start_j + batch_size
            xb = X[start_j:end_j]
            yb = y[start_j:end_j]

            y_pred = model(xb)
            loss = criterion(y_pred, yb, categories, periodicity)
            batch_loss.append((torch.abs(loss)).detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(inputs=xb, layers=list(model.children()))

        losses.append(sum(batch_loss) / len(batch_loss))
        if i % 10 == 9:
            print(f"Epoch {i} loss is {losses[-1]}")
        y_pred = model.predict(X)
        scores.append(accuracy(y_pred.squeeze(), y))

        # Logger.current_logger().report_scalar(
        #     "Loss/Acc", "Loss", iteration=i, value=losses[-1]
        # )
        # writer.add_scalar("Loss", losses[-1], i)
        # Logger.current_logger().report_scalar(
        #     "Loss/Acc", "Acc", iteration=i, value=scores[-1]
        # )
        # writer.add_scalar("Accuracy", scores[-1], i)

        # for key in model_dict:
        #     for key_layer in model_dict[key]:
        #         if key_layer in ["weights", "bias"]:
        #             log_label = str(key) + "_" + str(key_layer)
        #             log_label.replace(" ", "")
        #             writer.add_histogram(
        #                 log_label + "_real", model_dict[key][key_layer].real, i
        #             )
        #             writer.add_histogram(
        #                 log_label + "_imag", model_dict[key][key_layer].imag, i
        #             )
        #             writer.add_histogram(
        #                 log_label + "_mag", torch.abs(model_dict[key][key_layer]), i
        #             )
        #             writer.add_histogram(
        #                 log_label + "_angle", torch.angle(model_dict[key][key_layer]), i
        #             )

        # writer.add_histogram("distribution centers", x + n_iter, i)
        if scores[-1] > acc_best:
            acc_best = scores[-1]
            torch.save(model.state_dict(), PATH)

    # writer.close()
    return losses, scores

# %%
model = Model(categories=categories, periodicity=periodicity)
# criterion = ComplexMSELoss.apply
criterion = ComplexMSE_adjusted_error.apply
# clip_angle_value = 10000000
# optimizer = ECL(model.parameters(), lr=lr, clip_angle_value=clip_angle_value)
optimizer = ECL(model.parameters(), lr=lr)

# %%
# task = Task.init(
#     project_name="mlmvn",
#     task_name="SDD-mlmvn-[48-100-11]",
#     tags=["mlmvn", "SDD", "single_run"],
# )
# writer = SummaryWriter()

#  capture a dictionary of hyperparameters with config
config_dict = {
    "learning_rate": 1,
    "epochs": epochs,
    "batch_size": batch_size,
    "optim": "ECL",
    "categories": categories,
    "periodicity": periodicity,
    "layer": "[48-100-11]",
}
# task.connect(config_dict)

# %%
x_train, x_valid, y_train, y_valid = get_splitted_data(X, y, neuronCats)

losses, scores = fit(
    model,
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    categories=categories,
    periodicity=periodicity,
)

model.load_state_dict(torch.load(PATH))

y_pred = model.predict(x_train)
acc = accuracy(y_pred.squeeze(), y_train)
print("Train Acc.: ", acc)
# Logger.current_logger().report_single_value(
#     name="Train Acc.",
#     value=acc,
# )

y_pred = model.predict(x_valid)
acc = accuracy(y_pred.squeeze(), y_valid)
print("Val Acc.: ", acc)
# Logger.current_logger().report_single_value(
#     name="Val Acc.",
#     value=acc,
# )
print(classification_report(y_valid, y_pred.detach().numpy(), zero_division=0))

# %%
# task.mark_completed()
# task.close()
