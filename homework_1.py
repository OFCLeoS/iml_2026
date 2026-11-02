# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt  # making plots
import numpy as np  # array processing
import pandas as pd  # dataframes, reading files

import iml  # helper functions and classes (see iml/)

# %% [markdown]
# **Group members:** [your names]

# %% [markdown]
# ## Part 2: Linear Regression

# %% [markdown]
# ### (a)
# > Consider the linear model, defined by parameters $(\theta_0, \theta_1)$,
# > which predicts $$
# > \hat y(i) = \theta_0 + \theta_1 x(i)
# > $$ for each index
# > $i.$ Express the gradient $\nabla L$ of mean squared error as a function of
# > the parameter vector $\theta$ and the datapoints $(x(i), y(i)).$


x, y = iml.load_powerplant()
# print(x, y)


def predict_linear(theta, x):
    """
    Compute predictions for:
        y_hat[i] = theta[0] + theta[1] * x[i]
    """
    theta0 = theta[0]
    theta1 = theta[1]

    # TODO: calculate theta0 + theta1 times every element of x
    y_hat = theta0 + theta1 * x
    return y_hat


def mean_squared_error(theta, x, y):
    """
    Compute:
        L = (1 / N) * sum((y_hat[i] - y[i])**2)
    """
    y_hat = predict_linear(theta, x)
    residuals = y_hat - y  # TODO: prediction minus target
    loss = (1 / len(y)) * (residuals**2)  # TODO: mean of squared residuals

    return loss


def mse_gradient(theta, x, y):
    """
    Return the gradient as a NumPy array with shape (2,):

        gradient[0] = dL/dtheta_0
        gradient[1] = dL/dtheta_1
    """
    n = len(x)
    y_hat = predict_linear(theta, x)
    residuals = y_hat - y  # TODO

    dL_dtheta0 = (2 / n) * np.sum(
        residuals
    )  # TODO: sum the contribution from each datapoint
    dL_dtheta1 = (2 / n) * np.sum(
        residuals * x
    )  # TODO: determine the extra factor introduced by theta_1

    gradient = np.array(
        [
            dL_dtheta0,
            dL_dtheta1,
        ]
    )

    return gradient


# %% [markdown]
# ### (b)
# > Extend the `SimpleRegressor` class from `iml/simple_regressor.py`,
# > implementing the method `fit(self, x, y, lr, steps)` to implement gradient
# > descent on mean squared error. Run `fit()` on the _unnormalized_ data
# > vectors `x` and `y`, and show the training process over 20 iterations with
# > `SimpleRegressor.show_training()`. Find a learning rate that makes the loss
# > decrease.

from iml import SimpleRegressor


class GradientDescentRegressor(SimpleRegressor):
    def __init__(self):
        pass

    def fit(self, x, y, lr: float, steps: int = 20) -> None:
        self.weights_ = np.zeros(2)

        self.x_train_ = x.copy()
        self.y_train_ = y.copy()
        self.y_predicted_ = np.zeros((steps + 1, y.shape[0]))
        self.y_predicted_[0] = self.predict(x)

        for i in range(steps):
            residuals = self.predict(x) - y

            # Gradient of MSE with respect to [intercept, slope]
            gradient = np.array(
                [
                    2 * residuals.mean(),
                    2 * np.mean(residuals * x),
                ]
            )

            self.weights_ -= lr * gradient
            self.y_predicted_[i + 1] = self.predict(x)


model = GradientDescentRegressor()
model.fit(x, y, lr=0.0001, steps=20)
model.show_training()


# %% [markdown]
# ### (c)
# > Now, normalize your data in some way and run `fit()` and `show_training()`
# > again, ensuring that you converge to an acceptable model in 20 iterations.


x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()

x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std

model = GradientDescentRegressor()
model.fit(x_norm, y_norm, lr=0.3, steps=20)
model.show_training()
