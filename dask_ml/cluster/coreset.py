import inspect
import logging

import dask.array as da
import numpy as np
from sklearn.base import TransformerMixin

from .._utils import copy_learned_attributes
from ..utils import _timer
from ..wrappers import ParallelPostFit

logger = logging.getLogger(__name__)


def lightweight_coresets(X, m, *, gen=da.random.RandomState()):
    """
    Parameters
    ----------
    X : dask.array, shape = [n_samples, n_features]
        input dask arrat to be sampled
    m : int
        number of samples to pick from `X`

    gen: da.random.RandomState
        random state to use for sampling
    """
    dists = da.power(X - X.mean(), 2).sum(axis=1)
    q = 0.5 / X.shape[0] + 0.5 * dists / dists.sum()
    indices = gen.choice(X.shape[0], size=m, p=q, replace=True)
    w_lwcs = 1.0 / (m * q[indices])
    X_lwcs = X[indices, :]
    return X_lwcs, w_lwcs


class Coreset(ParallelPostFit, TransformerMixin):
    """
    Coreset sampling implementation

    A Coreset is a small set of points that approximates the shae of a larger point set.
    A clustering algorithm can be applied on the selected subset of points.

    Parameters
    ----------
    n_clusters : int, default 8
        Number of clusters to end up with

    m : int, default None
        number of points to select to form a coreset
        if estimator has a `n_clusters` or `n_components` attributes,
        `m` will be set to `(n_clusters|n_components)` * `X.shape[1]` / `eps` ** 2
        when calling `.fit`

    random_state : int, optional
    """

    def __init__(self, estimator, m=None, *, eps=0.05, random_state=None):
        if m is None:
            k = getattr(estimator, "n_clusters", None) or getattr(
                estimator, "n_components", None
            )
            if not k or not isinstance(k, int):
                raise ValueError(
                    """`m` is None, `estimator` must have
                    an attribute in (n_clusters, n_components)"""
                )
            self.k = k

        self.m = m
        self.eps = eps
        self.estimator = estimator
        self.random_state = da.random.RandomState(random_state)

    def fit(self, X, y=None, **kwargs):
        if self.k is not None and self.m is None:
            m = (X.shape[1] * self.k) / (self.eps ** 2)
            self.m = np.ceil(m)
        if self.m > X.shape[0]:
            logger.warning(
                f"""
                Number of points to sample ({self.m}) higher
                than input dimension ({X.shape[0]}),
                forcing reduction to {X.shape[0] * 0.05}
            """
            )
            self.m = X.shape[0] * 0.05

        print(f"sampling {self.m} points out of {X.shape[0]}")

        logger.info("Starting sampling")
        with _timer("sampling", _logger=logger):
            Xcs, weights = lightweight_coresets(X, self.m)
            Xcs = Xcs.compute()

        logger.info("Starting fit")
        with _timer("fit", _logger=logger):
            if "sample_weights" in inspect.signature(self.estimator.fit).parameters:
                kwargs["sample_weights"] = weights
            updated_est = self.estimator.fit(Xcs, y, **kwargs)

        # Copy over learned attributes
        copy_learned_attributes(updated_est, self)
        # return self  TODO
        return self
