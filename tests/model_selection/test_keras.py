import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from distributed.utils_test import gen_cluster
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import RandomizedSearchCV

from dask_ml.model_selection import IncrementalSearchCV

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("scikeras")

import tensorflow as tf
from tensorflow.keras.datasets import mnist as keras_mnist
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from dask_ml.wrappers import KerasClassifier, KerasRegressor, _BaseKerasWrapper


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
    X, y = mnist()
    assert X.ndim == 2 and X.shape[-1] == 784
    assert y.ndim == 1 and len(X) == len(y)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)

    model = KerasClassifier(build_fn=_keras_build_fn, lr=0.1)
    params = {"lr": loguniform(1e-3, 1e-1)}

    search = IncrementalSearchCV(
        model, params, max_iter=3, n_initial_parameters=5, decay_rate=None
    )
    yield search.fit(X, y)
    assert search.best_score_ >= 0

    # Make sure the model trains, and scores aren't constant
    scores = {
        ident: [h["score"] for h in hist]
        for ident, hist in search.model_history_.items()
    }
    assert all(len(hist) == 3 for hist in scores.values())
    nuniq_scores = [pd.Series(v).nunique() for v in scores.values()]
    assert max(nuniq_scores) > 1
