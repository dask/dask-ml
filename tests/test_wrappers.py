import pickle
from typing import Tuple

import numpy as np
import pytest
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from sklearn.datasets import make_classification, make_regression
from distributed.utils_test import gen_cluster
from scipy.stats import loguniform, uniform
from tensorflow.keras.datasets import mnist as keras_mnist
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from skorch import NeuralNetClassifier

from dask_ml.wrappers import (
    KerasClassifier,
    KerasRegressor,
    PyTorchClassifier,
    PyTorchRegressor,
)
from dask_ml.model_selection import IncrementalSearchCV
from sklearn.model_selection import RandomizedSearchCV


def mnist() -> Tuple[np.ndarray, np.ndarray]:
    (X_train, y_train), _ = keras_mnist.load_data()
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_train = X_train.astype("float32")
    X_train /= 255
    assert all(isinstance(x, np.ndarray) for x in [X_train, y_train])
    return X_train, y_train


def _keras_build_fn(optimizer="rmsprop", lr=0.01, kernel_initializer="glorot_uniform"):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer=kernel_initializer))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer=kernel_initializer))
    model.add(Activation("softmax"))  # This special "softmax" a

    opt = optimizer
    if optimizer == "SGD":
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def test_keras():
    X, y = mnist()
    assert X.shape[-1] == 784
    assert y.ndim == 1
    assert len(X) == len(y)
    model = KerasClassifier(build_fn=_keras_build_fn)
    model.partial_fit(X, y)


class ShallowNet(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.layer1 = nn.Linear(n_features, 1)

    def forward(self, x):
        return F.relu(self.layer1(x))


@gen_cluster(client=True)
def test_pytorch(c, s, a, b):
    n_features = 10
    clf = PyTorchRegressor(
        module=ShallowNet,
        module__n_features=n_features,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        optimizer__lr=0.1,
        batch_size=64,
    )
    from sklearn.base import clone
    clf2 = clone(clf)
    assert clf.callbacks == None
    assert clf.warm_start == False
    assert clf.train_split is None
    assert clf.max_epochs == 1

    params = {"optimizer__lr": loguniform(1e-3, 1e0)}
    X, y = make_regression(n_samples=100, n_features=n_features)
    X = X.astype("float32")
    y = y.astype("float32").reshape(-1, 1)
    search = IncrementalSearchCV(clf, params, max_iter=5, decay_rate=None)
    yield search.fit(X, y)
    assert search.best_score_ >= 0

def test_pytorch_doc():
    import torch.optim as optim
    import torch.nn as nn
    from dask_ml.wrappers import PyTorchRegressor
    import torch

    class ShallowNet(nn.Module):
        def __init__(self, n_features=5):
            super().__init__()
            self.layer1 = nn.Linear(n_features, 1)
        def forward(self, x):
            return torch.sign(self.layer1(x))

    model = PyTorchRegressor(
        module=ShallowNet,
        module__n_features=200,
        optimizer=optim.SGD,
        optimizer__lr=0.1,
        batch_size=64,
    )
    from sklearn.datasets import make_classification
    X, y = make_classification(n_features=200, n_samples=400, n_classes=2)
    X = X.astype("float32")
    y = y.astype("float32").reshape(-1, 1)
    model.partial_fit(X, y)
