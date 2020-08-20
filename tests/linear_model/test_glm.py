import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq
from dask_glm.regularizers import Regularizer
from sklearn.pipeline import make_pipeline

from dask_ml.datasets import make_classification, make_counts, make_regression
from dask_ml.linear_model import LinearRegression, LogisticRegression, PoissonRegression
from dask_ml.linear_model.utils import add_intercept
from dask_ml.model_selection import GridSearchCV


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def solver(request):
    """Parametrized fixture for all the solver names"""
    return request.param


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def regularizer(request):
    """Parametrized fixture for all the regularizer names"""
    return request.param


class DoNothingTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X

    def get_params(self, deep=True):
        return {}


X, y = make_classification(chunks=50)


def test_lr_init(solver):
    LogisticRegression(solver=solver)


def test_pr_init(solver):
    PoissonRegression(solver=solver)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit(fit_intercept, solver):
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)


@pytest.mark.parametrize(
    "solver", ["admm", "newton", "lbfgs", "proximal_grad", "gradient_descent"]
)
def test_fit_solver(solver):
    import dask_glm
    from distutils.version import LooseVersion

    if LooseVersion(dask_glm.__version__) <= "0.2.0":
        pytest.skip("FutureWarning for dask config.")

    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(solver=solver)
    lr.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lm(fit_intercept):
    X, y = make_regression(n_samples=100, n_features=5, chunks=50)
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_big(fit_intercept):
    X, y = make_classification(chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.decision_function(X)
    lr.predict(X)
    lr.predict_proba(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_poisson_fit(fit_intercept):
    X, y = make_counts(n_samples=100, chunks=500)
    pr = PoissonRegression(fit_intercept=fit_intercept)
    pr.fit(X, y)
    pr.predict(X)
    pr.get_deviance(X, y)
    if fit_intercept:
        assert pr.intercept_ is not None


def test_in_pipeline():
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    pipe.fit(X, y)


def test_gridsearch():
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    grid = {"logisticregression__C": [1000, 100, 10, 2]}
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    search = GridSearchCV(pipe, grid, cv=3)
    search.fit(X, y)


def test_add_intercept_dask_dataframe():
    X = dd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}), npartitions=2)
    result = add_intercept(X)
    expected = dd.from_pandas(
        pd.DataFrame(
            {"intercept": [1, 1, 1], "A": [1, 2, 3]}, columns=["intercept", "A"]
        ),
        npartitions=2,
    )
    assert_eq(result, expected)

    df = dd.from_pandas(pd.DataFrame({"intercept": [1, 2, 3]}), npartitions=2)
    with pytest.raises(ValueError):
        add_intercept(df)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_unknown_chunks_ok(fit_intercept):
    # https://github.com/dask/dask-ml/issues/145
    X = dd.from_pandas(pd.DataFrame(np.random.uniform(size=(10, 5))), 2).values
    y = dd.from_pandas(pd.Series(np.random.uniform(size=(10,))), 2).values

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y)


def test_add_intercept_unknown_ndim():
    X = dd.from_pandas(pd.DataFrame(np.ones((10, 5))), 2).values
    result = add_intercept(X)
    expected = np.ones((10, 6))
    da.utils.assert_eq(result, expected)


def test_add_intercept_raises_ndim():
    X = da.random.uniform(size=10, chunks=5)

    with pytest.raises(ValueError) as m:
        add_intercept(X)

    assert m.match("'X' should have 2 dimensions")


def test_add_intercept_raises_chunks():
    X = da.random.uniform(size=(10, 4), chunks=(4, 2))

    with pytest.raises(ValueError) as m:
        add_intercept(X)

    assert m.match("Chunking is only allowed")


def test_lr_score():
    X = da.from_array(np.arange(1000).reshape(1000, 1))
    lr = LinearRegression()
    lr.fit(X, X)
    assert lr.score(X, X) == pytest.approx(1, 0.001)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_dataframe_warns_about_chunks(fit_intercept):
    rng = np.random.RandomState(42)
    n, d = 20, 5
    kwargs = dict(npartitions=4)
    X = dd.from_pandas(pd.DataFrame(rng.uniform(size=(n, d))), **kwargs)
    y = dd.from_pandas(pd.Series(rng.choice(2, size=n)), **kwargs)
    clf = LogisticRegression(fit_intercept=fit_intercept)
    msg = "does not support dask dataframes.*might be resolved with"
    with pytest.raises(TypeError, match=msg):
        clf.fit(X, y)
    clf.fit(X.values, y.values)
    clf.fit(X.to_dask_array(), y.to_dask_array())
    clf.fit(X.to_dask_array(lengths=True), y.to_dask_array(lengths=True))
