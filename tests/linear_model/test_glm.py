import dask.dataframe as dd
import dask_searchcv as dcv
import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq
from dask_glm.regularizers import Regularizer
from sklearn.pipeline import make_pipeline

from dask_ml.datasets import make_classification, make_counts, make_regression
from dask_ml.linear_model import (LinearRegression, LogisticRegression,
                                  PoissonRegression)
from dask_ml.linear_model.utils import add_intercept


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def solver(request):
    """Parametrized fixture for all the solver names"""
    return request.param


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def regularizer(request):
    """Parametrized fixture for all the regularizer names"""
    return request.param


class DoNothingTransformer(object):
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


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_fit(fit_intercept, solver):
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)


@pytest.mark.parametrize('solver', ['admm', 'newton', 'lbfgs',
                                    'proximal_grad', 'gradient_descent'])
def test_fit_solver(solver):
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(solver=solver)
    lr.fit(X, y)


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_lm(fit_intercept):
    X, y = make_regression(n_samples=100, n_features=5, chunks=50)
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_big(fit_intercept):
    X, y = make_classification(chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize('fit_intercept', [True, False])
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
    grid = {
        'logisticregression__C': [1000, 100, 10, 2]
    }
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    search = dcv.GridSearchCV(pipe, grid, cv=3)
    search.fit(X, y)


def test_add_intercept_dask_dataframe():
    X = dd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}), npartitions=2)
    result = add_intercept(X)
    expected = dd.from_pandas(pd.DataFrame({"intercept": [1, 1, 1],
                                            "A": [1, 2, 3]},
                                           columns=['intercept', 'A']),
                              npartitions=2)
    assert_eq(result, expected)

    df = dd.from_pandas(pd.DataFrame({"intercept": [1, 2, 3]}), npartitions=2)
    with pytest.raises(ValueError):
        add_intercept(df)
