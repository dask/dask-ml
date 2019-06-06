import numpy as np
from distributed.utils_test import gen_cluster  # noqa: F401
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
import pytest

from dask_ml.model_selection import SuccessiveHalvingSearchCV
from dask_ml.model_selection._successive_halving import _get_n_initial_calls, _get_max_iter


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


@pytest.mark.parametrize("r", [3, 9, 27])
@pytest.mark.parametrize("n", [9, 27, 81])
def test_sha_max_iter(n, r):
    @gen_cluster(client=True)
    def _test_sha_max_iter(c, s, a, b):
        model = SGDClassifier(tol=1e-3)
        params = {"alpha": np.logspace(-3, 0, num=1000)}
        eta = 3
        search = SuccessiveHalvingSearchCV(
            model, params, n_initial_parameters=n, max_iter=_get_max_iter(n, r, eta)
        )

        X, y = make_classification()
        yield search.fit(X, y, classes=np.unique(y))

        calls = {v["partial_fit_calls"] for v in search.cv_results_} - {1}
        assert min(calls) == r

    if (n != 81 and r < n) or (n == 81 and r == 3):
        _test_sha_max_iter()
