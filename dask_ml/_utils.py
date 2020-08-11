import time

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def draw_seed(random_state, low, high=None, size=None, dtype=None, chunks=None):
    kwargs = {"size": size}
    if chunks is not None:
        kwargs["chunks"] = chunks

    seed = random_state.randint(low, high, **kwargs)
    if dtype is not None and isinstance(seed, (da.Array, np.ndarray)):
        seed = seed.astype(dtype)

    return seed


class ConstantFunction(BaseEstimator):
    def __init__(self, value=0, sleep=0, **kwargs):
        self.value = value
        self._partial_fit_called = False
        self.sleep = sleep
        super(ConstantFunction, self).__init__(**kwargs)

    def _fn(self):
        return self.value

    def partial_fit(self, X, y=None, **kwargs):
        time.sleep(self.sleep)
        self._partial_fit_called = True

        # Mirroring sklearn's SGDClassifier epoch counting
        if not hasattr(self, "t_"):
            self.t_ = 1
        self.t_ += X.shape[0]
        self.coef_ = X[0]
        return self

    def score(self, *args, **kwargs):
        return self._fn()

    def fit(self, *args):
        return self
