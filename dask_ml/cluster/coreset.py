import inspect
import logging

import dask.array as da
import dask.dataframe as dd
import numpy as np
from sklearn.base import TransformerMixin, clone

from .._utils import copy_learned_attributes
from ..utils import _timer, check_array
from ..wrappers import ParallelPostFit

logger = logging.getLogger(__name__)


def lightweight_coresets(X, m, *, gen=da.random.RandomState()):
    """
    Parameters
    ----------
    X : dask.array, shape = [n_samples, n_features]
        input dask array to be sampled
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


def get_m(X, k, eps, mode="hard"):
    """
    Returns the coreset size, i.e the number of data points to be sampled
    from the original set of points
    See Theorem 2 from `Scalable k-Means Clustering via Lightweight Coresets`

    The resulting coreset is a (eps, k)-lightweight coreset of X

    Parameters
    ----------
    X: dask.array
        input data. We aim at finding minimum coreset size to be sampled from `X`
    mu: float
        average value for the input data
    k: int
        number of cluster k
    eps: float
        between 0 and 1


    Returns
    -------
    m: int
        Number of points to sample
        For this number, the set `C` is a (`eps`, `k`)-lightweight coreset of `X`

    Notes
    -----
    The `delta` parameter from the original paper is not used.
    In practice most of the time it vanishes,
    as a log is applied to its inverse, before being summed to big values
    (values which depend on the data size | number of cluster)
    """
    X_m, d = X.shape
    if hasattr(X_m, "compute"):
        X_m = X_m.compute()

    if mode == "hard":  # hard clustering
        numerator = d * k * np.log(k)
    elif mode == "soft":  # soft clustering
        numerator = (d ** 2) * (k ** 2)
    else:
        raise ValueError("`mode` should be in (hard|soft)")
    m = np.ceil(numerator / eps)
    if m >= X_m:
        _m = np.ceil((d ** 2) * (k ** 2))
        logger.warning(
            f"""
            Number of points to sample ({m}) higher
            than input dimension ({d}),
            forcing reduction to {_m}
        """
        )
        m = _m
    return m


class Coreset(ParallelPostFit, TransformerMixin):
    """Coreset sampling implementation

    Parameters
    ----------
    estimator : Estimator
        The underlying estimator to be fitted.

    eps: float, default=0.05
        For k cluster, the coreset is guaranteed to be a (`eps`, `k`)
        coreset of the original data

        `eps` must be greater or equal to 0.05
        (<= 5% difference in the discretization error).

    m : int, default None
        Number of points to select to form a coreset

        If it is `None` and the estimator has a `n_clusters` or `n_components`
        attributes, `m` will atomatically be set depending on
        `n_clusters|n_components`, `eps`and the input data when calling `.fit`

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    - Scalable k-Means Clustering via Lightweight Coresets, 2018
      Olivier Bachem, Mario Lucic, Andreas Krause
      https://arxiv.org/pdf/1702.08248.pdf

    Notes
    -----
    ``A Coreset is a small set of points that approximates
    the shape of a larger set of points``.

    A clustering algorithm can be applied on the selected subset of points.

    Formally, a weighted set `C` is an (`eps`, `k`)-coreset for
    some input data X if for any set of cluster centers `Q` (with `|Q| <= k`)
    the quantization error computed via `Q` on `X` and
    the quantization error computer via `Q` on `C` have at most an
    `eps` relative difference.

    """

    def __init__(self, estimator, *, eps=0.05, delta=0.01, m=None, random_state=None):
        if not (0 < delta < 1):
            raise ValueError("`delta` both should be a float between 0 and 1")
        if not (0.05 <= eps < 1):
            raise ValueError("`eps` should be a float between 0.05 and 1")
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
        self.estimator = clone(estimator)
        self.random_state = da.random.RandomState(random_state)

    def fit(self, X, y=None, **kwargs):
        if isinstance(X, dd.DataFrame):
            X = X.to_dask_array(lengths=True)  # if Dask.Dataframe
        X = check_array(X, accept_dask_dataframe=False)
        if self.m is None:
            self.m = get_m(X, self.k, self.eps)

        logger.info(f"sampling {self.m} points out of {X.shape[0]}")

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
        ParallelPostFit.__init__(self, estimator=updated_est)
        return self

    # TODO : partial fit ?
