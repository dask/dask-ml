import inspect

import dask.array as da
import dask.dataframe as dd
import pytest
import sklearn.base
import sklearn.linear_model

from dask_ml.datasets import make_regression, make_classification
from dask_ml.utils import assert_estimator_equal

import dask_ml.iid.cluster
import dask_ml.iid.covariance
import dask_ml.iid.decomposition
import dask_ml.iid.ensemble
import dask_ml.iid.gaussian_process
import dask_ml.iid.linear_model
import dask_ml.iid.manifold
import dask_ml.iid.mixture
import dask_ml.iid.neighbors
import dask_ml.iid.neural_network
import dask_ml.iid.preprocessing
import dask_ml.iid.semi_supervised
import dask_ml.iid.svm
import dask_ml.iid.tree


skips = {
    # Deprecated, and doesn't use wraps so no signature
    'RandomizedLasso',
    'RandomizedLogisticRegression',
    'RandomizedPCA',
    # 'errors' don't match
    'DictionaryLearning',
    # Different __init__
    'SparseCoder',
    # Abstract
    'BaseEnsemble',  # TODO: see if you can filter these. Why are they in all?
    # TODO
    'VotingClassifier',
    # Bug in scikit-learn?
}


@pytest.fixture(params=(
    dask_ml.iid.cluster._models +
    dask_ml.iid.covariance._models +
    dask_ml.iid.decomposition._models +
    dask_ml.iid.ensemble._models +
    dask_ml.iid.gaussian_process._models +
    dask_ml.iid.linear_model._models +
    dask_ml.iid.manifold._models +
    dask_ml.iid.mixture._models +
    dask_ml.iid.neighbors._models +
    dask_ml.iid.neural_network._models +
    dask_ml.iid.preprocessing._models +
    dask_ml.iid.semi_supervised._models +
    dask_ml.iid.svm._models +
    dask_ml.iid.tree._models
))
def model(request):
    return request.param


def test_linear_regression():
    X, y = make_regression(n_samples=100, chunks=50, random_state=0)

    real = sklearn.linear_model.LinearRegression()
    fake = dask_ml.iid.linear_model.LinearRegression()

    real.fit(X[:50], y[:50])
    fake.fit(X, y)

    assert_estimator_equal(fake, real, tol=1e-5)
    assert isinstance(fake.predict(X), type(X))

    X, y = make_regression(n_samples=100, chunks=50, random_state=0)
    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)

    real = sklearn.linear_model.LinearRegression()
    fake = dask_ml.iid.linear_model.LinearRegression()

    real.fit(X.get_partition(0), y.get_partition(0))
    fake.fit(X, y)

    assert_estimator_equal(fake, real, tol=1e-5)
    assert isinstance(fake.predict(X), da.Array)


def test_all(model):
    if model.__name__ in skips:
        pytest.skip(model.__name__)

    kwargs = {}
    if 'random_state' in set(inspect.getfullargspec(model).args):
        kwargs['random_state'] = 0

    if isinstance(model, sklearn.base.RegressorMixin):
        X, y = make_regression(n_samples=100, chunks=50, random_state=0)
    else:
        X, y = make_classification(n_samples=100, chunks=50, random_state=0)

    real = model(**kwargs)
    fake = getattr(getattr(dask_ml.iid, model.__module__.split('.')[1]),
                   model.__name__)(**kwargs)
    with pytest.warns(None):
        # We don't care about any deprecation warnings
        try:
            real.fit(X[:50], y[:50])
        except Exception as e:
            pytest.skip(e.args[0])

        fake.fit(X, y)

        # Bug in sklearn for SVR.n_support?
        assert_estimator_equal(fake, real, atol=1e-5, exclude=['n_support_'])

    if hasattr(real, 'predict'):
        assert isinstance(fake.predict(X), type(X))
    else:
        has_it = hasattr(fake, 'predict')
        if has_it:
            # Some attrs depend on specific values
            with pytest.raises(AttributeError):
                fake.predict(X)

    if hasattr(real, 'predict_proba'):
        assert isinstance(fake.predict_proba(X), type(X))
    else:
        has_it = hasattr(fake, 'predict_proba')

        if has_it:
            # Some attrs depend on specific values
            with pytest.raises(AttributeError):
                fake.predict_proba(X)
