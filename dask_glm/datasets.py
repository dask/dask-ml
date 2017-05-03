import numpy as np
import dask.array as da
from dask_glm.utils import exp


def make_classification(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                        chunksize=100):
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    y = da.random.random(z0.shape, chunks=(chunksize,)) < 1 / (1 + da.exp(-z0))
    return X, y


def make_regression(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                    chunksize=100):
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    y = da.random.random(z0.shape, chunks=(chunksize,))
    return X, y


def make_poisson(n_samples=1000, n_features=100, n_informative=2, scale=1.0,
                 chunksize=100):
    X = da.random.normal(0, 1, size=(n_samples, n_features),
                         chunks=(chunksize, n_features))
    informative_idx = np.random.choice(n_features, n_informative)
    beta = (np.random.random(n_features) - 1) * scale
    z0 = X[:, informative_idx].dot(beta[informative_idx])
    rate = exp(z0)
    y = da.random.poisson(rate, size=1, chunks=(chunksize,))
    return X, y
