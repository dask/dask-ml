import numpy as np
from distributed.utils_test import cluster, gen_cluster, loop  # noqa: F401
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection import SuccessiveHalvingSearchCV, HyperbandSearchCV


@gen_cluster(client=True)
def test_basic_successive_halving(c, s, a, b):
    model = SGDClassifier(tol=1e-3)
    params = {"alpha": np.logspace(-3, 0, num=1000)}
    n, r = 10, 5
    search = SuccessiveHalvingSearchCV(model, params, n, r)

    X, y = make_classification()
    yield search.fit(X, y, classes=np.unique(y))
    assert search.best_score_ > 0
    assert isinstance(search.best_estimator_, SGDClassifier)
class LinearFunction(SGDClassifier):
    def __init__(self, intercept=0, slope=0, **kwargs):
        self.intercept = intercept
        self.slope = slope
        self._pf_calls = 0
        super().__init__(**kwargs)

    def partial_fit(self, *args, **kwargs):
        self._pf_calls += 1
        return self

    def _fn(self, x):
        return self.slope * x + self.intercept

    def score(self, *args, **kwargs):
        return self._fn(self._pf_calls)


@gen_cluster(client=True, timeout=5000)
def test_patience(c, s, a, b):
    model = LinearFunction()
    params = {"slope": [1] * 244}
    max_iter = 243

    hyperband = HyperbandSearchCV(model, params, max_iter=max_iter)
    bracket = hyperband.metadata()["brackets"]["bracket=2"]
    succ_halv = bracket["SuccessiveHalvingSearchCV params"]
    assert succ_halv == {
        "aggressiveness": 3,
        "limit": 3,
        "n_initial_parameters": 15,
        "resource": 27,
    }
    assert bracket == {
        "partial_fit_calls": 837,
        "bracket": 2,
        "models": 15,
        "iters": [1, 27, 81, 243],
        "SuccessiveHalvingSearchCV params": succ_halv,
    }

    search = SuccessiveHalvingSearchCV(
        model, params, patience=max_iter // 3, tol=1e-3, **succ_halv
    )
    yield search.fit(*make_classification())

    est_calls = {
        k: [vi["partial_fit_calls"] for vi in v]
        for k, v in search.estimator_history_.items()
    }
    max_calls = {k: max(calls) for k, calls in est_calls.items()}

    # Most trained model should be trained for 243 partial_fit calls
    assert max(max_calls.values()) == max_iter
