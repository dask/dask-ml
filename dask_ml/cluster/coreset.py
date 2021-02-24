from sklearn.base import BaseEstimator, TransformerMixin

import logging
import dask.array as da
import numpy as np
from ..utils import _timer
from .._utils import copy_learned_attributes
import inspect

logger = logging.getLogger(__name__)

def lightweight_coresets(X, m):
    """
    Parameters
    ----------
    X : dask.array, shape = [n_samples, n_features]
        input dask arrat to be sampled
    m : int
        number of samples to pick from `X`
    """
    dists = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
    q = 0.5 / X.shape[0] + 0.5 * (dists / dists.sum())
    idxs = da.random.choice(X.shape[0], size=m, p=q)
    weights = 1.0 / (m * q[idxs])
    return X[idxs, :], weights


class Coreset(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, m=None, k=None, eps=0.05):
        self.m = m
        self.eps = eps
        self.estimator = estimator

        if hasattr(estimator, "n_clusters"):  # eg. KMeans()
            self.k = estimator.n_clusters
            logger.info(f"k set to {self.k}")
        elif hasattr(estimator, "n_components"):  # eg. GaussianMixture
            self.k = estimator.n_components
            logger.info(f"k set to {self.k}")


    def fit(self, X, y=None, **kwargs):
        if self.k is not None and self.m is None:
            m = (X.shape[1] * self.k * np.log2(self.k)) / (self.eps ** 2)
            self.m = np.ceil(m)
        if self.m > X.shape[0]:
            logger.warning(f"""
                Number of points ({self.m}) to sample higher 
                than input dimension ({X.shape[0]}), forcing reduction to {X.shape[0] * 0.05}
            """)
            self.m = X.shape[0] * 0.05

        print(f"sampling {self.m} points out of {X.shape[0]}")

        logger.info("Starting sampling")
        with _timer("sampling", _logger=logger):
            Xcs, weights = lightweight_coresets(X, self.m)
            #Xcs *= weights.reshape((len(weights), 1))  # TODO weights must be fixed for that
            Xcs = Xcs.compute()

        #TODO check `init_params` to `kmeans` for GaussianMixture

        #TODO : use dask_ml.cluster.k_means.init for to init centroids as in
        # https://github.com/zalanborsos/coresets/blob/47896a68c79666496cf1ef1d2683bd76875fe013/coresets/k_means_coreset.py#L39
        logger.info("Starting fit")
        with _timer("fit", _logger=logger):
            if "sample_weights" in inspect.signature(self.estimator.fit).parameters:
                kwargs["sample_weights"] = weights
            updated_est = self.estimator.fit(Xcs, y, **kwargs)


        # Copy over learned attributes
        #copy_learned_attributes(updated_est, self)  TODO 
        # return self  TODO 
        return updated_est
