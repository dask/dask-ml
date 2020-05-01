import numpy as np
import pandas as pd
import pytest
from distributed.utils_test import gen_cluster  # noqa: F401
from scipy.optimize import curve_fit
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection import InverseDecaySearchCV


@gen_cluster(client=True)
def test_basic_inverse(c, s, a, b):
    # Most of the basics are tested through Hyperband (which relies on
    # successive halving)
    model = SGDClassifier(tol=1e-3)
    params = {"alpha": np.logspace(-3, 0, num=1000)}
    search = InverseDecaySearchCV(model, params, n_initial_parameters=5, max_iter=5)

    X, y = make_classification()
    yield search.fit(X, y, classes=np.unique(y))
    assert search.best_score_ > 0
    assert isinstance(search.best_estimator_, SGDClassifier)


@gen_cluster(client=True)
def test_inverse_decay(c, s, a, b):
    model = SGDClassifier(tol=1e-3)
    params = {"alpha": np.logspace(-3, 0, num=1000)}
    n_init = 10
    search = InverseDecaySearchCV(model, params, n_initial_parameters=n_init, max_iter=n_init)

    X, y = make_classification()
    yield search.fit(X, y, classes=np.unique(y))
    df = pd.DataFrame(search.history_)
    calls_per_model = df.groupby("model_id")["partial_fit_calls"].max()
    n_models = calls_per_model.value_counts()
    n_models.index.name = "n_calls"
    n_models.name = "n_models"

    # Get cumulative number of models condiered at each time step
    n_models = n_models.sort_index(ascending=False).cumsum()

    calls = n_models.index.values
    models = n_models.values

    def inv(x, initial):
        return initial / x

    popt, pcov = curve_fit(inv, calls, models)
    m_hat = inv(calls, *popt)
    m_hat = np.round(m_hat).astype(int)

    assert (m_hat != models).sum() <= 1 and len(models) == 6
    assert models.min() == 2


@pytest.mark.parametrize("fits_per_score", [1, 2, 4])
def test_fits_per_score(fits_per_score):
    @gen_cluster(client=True)
    def _test_fits_per_score(c, s, a, b):
        model = SGDClassifier(tol=1e-3)
        params = {"alpha": np.logspace(-3, 0, num=1000)}
        n_init = 10
        search = InverseDecaySearchCV(
            model, params, n_initial_parameters=n_init, fits_per_score=fits_per_score,
            max_iter=n_init,
        )

        X, y = make_classification()
        yield search.fit(X, y, classes=np.unique(y))
        best_id = search.cv_results_["model_id"][search.best_index_]
        hist = search.model_history_[best_id]
        df = pd.DataFrame(hist)
        waits = df.partial_fit_calls.diff().iloc[1:]
        assert np.nanmax(waits) == fits_per_score

    _test_fits_per_score()


@pytest.mark.parametrize("patience", [False, 3, 5])
def test_patience(patience):
    @gen_cluster(client=True)
    def _get_patience(c, s, a, b):
        model = SGDClassifier(tol=1e-3)
        params = {"alpha": np.logspace(-3, 0, num=1000)}
        n_init = 20
        # don't score the models at all if possible
        search = InverseDecaySearchCV(
            model,
            params,
            n_initial_parameters=n_init,
            patience=patience,
            fits_per_score=n_init // 2,
            max_iter=1 * n_init,
        )

        X, y = make_classification()
        yield search.fit(X, y, classes=np.unique(y))
        return search

    search = _get_patience()
    n_models = len(search.model_history_)
    best_id = search.cv_results_["model_id"][search.best_index_]
    hist = search.model_history_[best_id]
    df = pd.DataFrame(hist)
    waits = df.partial_fit_calls.diff()
    if patience:
        assert np.nanmax(waits) <= patience
    else:
        assert np.nanmax(waits) == n_models // 2  # b/c specified fits_per_score=10
