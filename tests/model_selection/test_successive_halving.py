import numpy as np
import pytest
from distributed.utils_test import gen_cluster  # noqa: F401
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from dask_ml._compat import DISTRIBUTED_2_5_0
from dask_ml.model_selection import SuccessiveHalvingSearchCV
from dask_ml.utils import ConstantFunction

pytestmark = pytest.mark.skipif(not DISTRIBUTED_2_5_0, reason="hangs")


@gen_cluster(client=True)
def test_basic_successive_halving(c, s, a, b):
    # Most of the basics are tested through Hyperband (which relies on
    # successive halving)
    model = SGDClassifier(tol=1e-3)
    params = {"alpha": np.logspace(-3, 0, num=1000)}
    n, r = 10, 5
    search = SuccessiveHalvingSearchCV(model, params, n, r)

    X, y = make_classification()
    yield search.fit(X, y, classes=np.unique(y))
    assert search.best_score_ > 0
    assert isinstance(search.best_estimator_, SGDClassifier)


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("n", [9, 22])
def test_sha_max_iter_and_metadata(n, r):
    # This test makes sure the number of partial fit calls is perserved
    # when

    # * n_initial_parameters and max_iter are specified
    # * n_initial_parameters is specified (but max_iter isn't)
    # * max_iter is specified (but n_initial_parameters isn't)

    # n_initial_parameters and max_iter are chosen to make sure the
    # successivehalving works as expected
    # (so only one model is obtained at the end, as per the last assert)

    @gen_cluster(client=True)
    def _test_sha_max_iter(c, s, a, b):
        model = SGDClassifier(tol=1e-3)
        params = {"alpha": np.logspace(-3, 0, num=1000)}
        search = SuccessiveHalvingSearchCV(
            model, params, n_initial_parameters=n, n_initial_iter=r
        )

        X, y = make_classification()
        yield search.fit(X, y, classes=np.unique(y))

        calls = set(search.cv_results_["partial_fit_calls"]) - {1}
        assert min(calls) == r

        # One model trained to completion
        assert (
            search.cv_results_["partial_fit_calls"] == max(calls)
        ).sum() < search.aggressiveness

        assert search.metadata == search.metadata_
        assert set(search.metadata.keys()) == {
            "partial_fit_calls",
            "n_models",
            "max_iter",
        }

    _test_sha_max_iter()


@gen_cluster(client=True)
def test_search_patience_infeasible_tol(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()

    search = SuccessiveHalvingSearchCV(
        model,
        params,
        patience=2,
        tol=np.nan,
        n_initial_parameters=20,
        n_initial_iter=4,
        max_iter=1000,
    )
    yield search.fit(X, y, classes=[0, 1])

    assert search.metadata_["partial_fit_calls"] == search.metadata["partial_fit_calls"]
    assert search.metadata_ == search.metadata
