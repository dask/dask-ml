import numpy as np
import pytest
from distributed.utils_test import gen_cluster  # noqa: F401
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection import SuccessiveHalvingSearchCV
from dask_ml.model_selection._successive_halving import (
    _get_max_iter,
    _get_n_initial_calls,
)


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


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("n", [9, 15])
def test_sha_max_iter(n, r):
    """
    This test makes sure the number of partial fit calls is perserved
    when

    * n_initial_parameters and max_iter are specified
    * n_initial_parameters is specified (but max_iter isn't)
    * max_iter is specified (but n_initial_parameters isn't)

    n_initial_parameters and max_iter are chosen to make sure the
    successivehalving works as expected
    (so only one model is obtained at the end, as per the last assert)

    """

    @gen_cluster(client=True)
    def _test_sha_max_iter(c, s, a, b):
        model = SGDClassifier(tol=1e-3)
        params = {"alpha": np.logspace(-3, 0, num=1000)}
        eta = 3
        max_iter = _get_max_iter(n, r, eta)
        search1 = SuccessiveHalvingSearchCV(
            model, params, n_initial_parameters=n, max_iter=max_iter
        )
        search2 = SuccessiveHalvingSearchCV(
            model, params, n_initial_parameters=n, max_iter=max_iter, n_initial_iter=r
        )
        search3 = SuccessiveHalvingSearchCV(
            model, params, n_initial_parameters=n, n_initial_iter=r
        )

        X, y = make_classification()
        yield search1.fit(X, y, classes=np.unique(y))
        yield search2.fit(X, y, classes=np.unique(y))
        yield search3.fit(X, y, classes=np.unique(y))

        for search in [search1, search2, search3]:
            calls = set(search.cv_results_["partial_fit_calls"]) - {1}
            assert min(calls) == r
            assert max(calls) == max_iter

        # Making sure models are called same number of times
        assert (
            np.sort(search1.cv_results_["partial_fit_calls"])
            == np.sort(search2.cv_results_["partial_fit_calls"])
        ).all()
        assert (
            np.sort(search1.cv_results_["partial_fit_calls"])
            == np.sort(search3.cv_results_["partial_fit_calls"])
        ).all()
        pf_calls = search1.cv_results_["partial_fit_calls"]
        assert (pf_calls == pf_calls.max()).sum() == 1

    if (n != 81 and r < n) or (n == 81 and r == 3):
        _test_sha_max_iter()
