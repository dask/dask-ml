import pytest
import dask

from dask_glm.estimators import LogisticRegression, LinearRegression, PoissonRegression
from dask_glm.datasets import make_classification, make_regression, make_poisson
from dask_glm.regularizers import Regularizer


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


X, y = make_classification()


def test_lr_init(solver):
    LogisticRegression(solver=solver)


def test_pr_init(solver):
    PoissonRegression(solver=solver)


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('is_sparse', [True, False])
def test_fit(fit_intercept, is_sparse):
    X, y = make_classification(n_samples=100, n_features=5, chunksize=10, is_sparse=is_sparse)
    lr = LogisticRegression(fit_intercept=fit_intercept, use_sparse_matrix=is_sparse)
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('is_sparse', [True, False])
def test_lm(fit_intercept, is_sparse):
    X, y = make_regression(n_samples=100, n_features=5, chunksize=10, is_sparse=is_sparse)
    lr = LinearRegression(fit_intercept=fit_intercept, use_sparse_matrix=is_sparse)
    lr.fit(X, y)
    lr.predict(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('is_sparse', [True, False])
def test_big(fit_intercept, is_sparse):
    with dask.config.set(scheduler='synchronous'):
        X, y = make_classification(is_sparse=is_sparse)
        lr = LogisticRegression(fit_intercept=fit_intercept, use_sparse_matrix=is_sparse)
        lr.fit(X, y)
        lr.predict(X)
        lr.predict_proba(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('is_sparse', [True, False])
def test_poisson_fit(fit_intercept, is_sparse):
    with dask.config.set(scheduler='synchronous'):
        X, y = make_poisson(is_sparse=is_sparse)
        pr = PoissonRegression(fit_intercept=fit_intercept, use_sparse_matrix=is_sparse)
        pr.fit(X, y)
        pr.predict(X)
        pr.get_deviance(X, y)
    if fit_intercept:
        assert pr.intercept_ is not None


def test_in_pipeline():
    from sklearn.pipeline import make_pipeline
    X, y = make_classification(n_samples=100, n_features=5, chunksize=10)
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    pipe.fit(X, y)


def test_gridsearch():
    from sklearn.pipeline import make_pipeline
    dcv = pytest.importorskip('dask_searchcv')

    X, y = make_classification(n_samples=100, n_features=5, chunksize=10)
    grid = {
        'logisticregression__lamduh': [.001, .01, .1, .5]
    }
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    search = dcv.GridSearchCV(pipe, grid, cv=3)
    search.fit(X, y)
