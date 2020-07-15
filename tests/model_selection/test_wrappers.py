import pickle
from typing import Tuple

import numpy as np
import pytest
from distributed.utils_test import gen_cluster
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification, make_regression
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning

from dask_ml.model_selection import IncrementalSearchCV

try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist as keras_mnist
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except:
    pass

try:
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from skorch import NeuralNetClassifier, NeuralNetRegressor
except:
    pass


def mnist() -> Tuple[np.ndarray, np.ndarray]:
    (X_train, y_train), _ = keras_mnist.load_data()
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_train = X_train.astype("float32")
    X_train /= 255
    Y_train = to_categorical(y_train, 10)
    return X_train, y_train


def _keras_build_fn(lr=0.01):
    layers = [
        Dense(512, input_shape=(784,), activation="relu"),
        Dense(10, input_shape=(512,), activation="softmax"),
    ]
    model = Sequential(layers)

    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


@gen_cluster(client=True)
def test_keras(c, s, a, b):
#  def test_keras():
    pytest.importorskip("tensorflow")
    pytest.importorskip("scikeras")

    X, y = mnist()
    assert X.ndim == 2 and X.shape[-1] == 784
    assert y.ndim == 1 and len(X) == len(y)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    model = KerasClassifier(build_fn=_keras_build_fn, lr=0.1)
    params = {"lr": loguniform(1e-3, 1e-1)}

    # SciKeras reformats some of the shapes; I think this warning comes from
    # that but am not sure
    # https://github.com/adriangb/scikeras/issues/19#issuecomment-658549923
    with pytest.warns(DataConversionWarning):
        m = model.partial_fit(X, y)
    assert m is model
    model2 = pickle.loads(pickle.dumps(model))

    search = RandomizedSearchCV(model, params)
    with pytest.warns(DataConversionWarning):
        search.fit(X, y, epochs=2)
    assert search.best_score_ >= 0

    search = IncrementalSearchCV(model, params, max_iter=5)
    with pytest.warns(DataConversionWarning):
        yield search.fit(X, y)
    assert search.best_score_ >= 0


class ShallowNet(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.layer1 = nn.Linear(n_features, 1)

    def forward(self, x):
        return F.relu(self.layer1(x))


@gen_cluster(client=True)
def test_pytorch(c, s, a, b):
    pytest.importorskip("torch")
    pytest.importorskip("skorch")

    n_features = 10
    defaults = {
        "callbacks": False,
        "warm_start": False,
        "train_split": None,
        "max_epochs": 1,
    }
    model = NeuralNetRegressor(
        module=ShallowNet,
        module__n_features=n_features,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        optimizer__lr=0.1,
        batch_size=64,
        **defaults,
    )

    model2 = clone(model)
    assert model.callbacks == False
    assert model.warm_start == False
    assert model.train_split is None
    assert model.max_epochs == 1

    params = {"optimizer__lr": loguniform(1e-3, 1e0)}
    X, y = make_regression(n_samples=100, n_features=n_features)
    X = X.astype("float32")
    y = y.astype("float32").reshape(-1, 1)
    search = IncrementalSearchCV(model2, params, max_iter=5, decay_rate=None)
    yield search.fit(X, y)
    assert search.best_score_ >= 0
