from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np


class SimpleRegression:
    def __init__(self):
        pass

    def predict(self, x):
        return self.weights_[0] + x[:, 0] * self.weights_[1]

    def fit(self, x, y, lr: float, steps: int = 20):
        self.weights_ = np.zeros(2)

        self.x_train_ = x.copy()
        self.y_train_ = y.copy()
        self.y_predicted_ = np.zeros((steps + 1, y.shape[0]))
        self.y_predicted_[0] = self.predict(x)

        for i in range(steps):
            # here: update self.weights_?
            self.y_predicted_[i + 1] = self.predict(x)

    def show_training(self):
        show_regression_training(self.x_train_, self.y_train_, self.y_predicted_)


def show_regression_training(x_train, y_train, y_predicted):
    """
    Given one-dimensional arrays x_train and y_train and a matrix y_predicted of predictions made by a model during optimization, graph both the data and the succession of model predictions.
    """
    fig, [loss_ax, graph_ax] = plt.subplots(1, 2, layout="tight")
    fig.set_size_inches(12, 5)

    # MSE loss
    loss_ax.set_title("MSE loss")
    loss_ax.set_box_aspect(1)

    loss = np.pow(y_predicted - y_train, 2).mean(axis=-1)
    loss_ax.plot(loss)
    loss_ax.set_ylabel("loss")
    loss_ax.set_xlabel("iteration")
    loss_ax.set_ylim(0, loss.max() * 1.1)

    # Intermediate models
    graph_ax.set_title("Intermediate models")
    graph_ax.scatter(x_train, y_train, label="data")

    iterations = np.arange(0, y_predicted.shape[0])
    if y_predicted.shape[0] > 1000:
        iterations = np.astype(np.linspace(0, y_predicted.shape[0] - 1, num=500), int)

    steps = y_predicted.shape[0]
    cmap = plt.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=steps - 1)
    for i in iterations:
        line = graph_ax.plot(x_train, y_predicted[i], c=cmap(norm(i)))
    line[0].set_label("model")  # pyright: ignore

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=graph_ax, label="iteration", location="right")

    plt.legend()
    plt.show(fig)
