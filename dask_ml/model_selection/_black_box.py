import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin


class BlackBox(BaseEstimator, MetaEstimatorMixin):
    """
    Fit an estimator with a subset of the data passed.

    This is most useful in cross validation searches because it treats the
    estimator as a black box.

    Parameters
    ----------
    estimator : BaseEstimator
        The base estimator to fit with data

    max_calls : int, optional
        The maximum number of times this estimator wil be called.

    """

    def __init__(self, estimator, max_calls=1, **kwargs):
        self.estimator = estimator
        self.estimator.set_params(**kwargs)

        self.max_calls = max_calls
        self._calls = 0

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def partial_fit(self, X, y=None):
        self._calls += 1
        frac = self._calls / self.max_calls
        n = X.shape[0]
        idx = np.random.permutation(n)[: int(n * frac)]

        self.estimator.fit(X[idx], y[idx])
        return self

    def score(self, X, y=None):
        return self.estimator.score(X, y)

    def get_params(self, **kwargs):
        return {"estimator": self.estimator, **self.estimator.get_params(**kwargs)}

    def set_params(self, **kwargs):
        est = kwargs.pop("estimator", None)
        if est is not None:
            self.estimator = est
        self.estimator.set_params(**kwargs)
        return self
