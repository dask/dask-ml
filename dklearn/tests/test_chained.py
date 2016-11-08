from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import dask.array as da
import dask.bag as db
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, SGDClassifier

import dklearn.matrix as dm
from dklearn.wrapped import Wrapped
from dklearn.chained import Chained

# Not fit estimators raise NotFittedError, but old versions of scikit-learn
# include two definitions of this error, which makes it hard to catch
# appropriately. Since it subclasses from `AttributeError`, this is good
# enough for the tests.
NotFittedError = AttributeError

sgd1 = SGDClassifier(penalty='l1')
sgd2 = SGDClassifier(penalty='l2')

iris = load_iris()
X_iris = iris.data[:, :2]
y_iris = iris.target


def test_Chained__init__():
    c1 = Chained(sgd1)
    c2 = Chained(sgd2)

    assert c1._name != c2._name
    assert Chained(sgd1)._name == c1._name

    with pytest.raises(TypeError):
        Chained("not an estimator")

    # No partial_fit
    lr = LogisticRegression()
    with pytest.raises(ValueError):
        Chained(lr)


def test_Chained_from_sklearn():
    c1 = Chained(sgd1)
    assert Chained.from_sklearn(sgd1)._name == c1._name
    assert Chained.from_sklearn(c1) is c1


def test_tokenize_Chained():
    c1 = Chained(sgd1)
    c2 = Chained(sgd2)
    assert tokenize(c1) == tokenize(c1)
    assert tokenize(c1) != tokenize(c2)
    assert tokenize(c1) != tokenize(Wrapped(sgd1))


def test_clone():
    c = Chained(sgd1)
    c2 = clone(c)
    assert isinstance(c2, Chained)
    assert c._name == c2._name
    assert c._est is not c2._est


def test__estimator_type():
    c = Chained(sgd1)
    assert c._estimator_type == sgd1._estimator_type


def test_get_params():
    c = Chained(sgd1)
    assert c.get_params() == sgd1.get_params()
    assert c.get_params(deep=False) == sgd1.get_params(deep=False)


def test_set_params():
    c = Chained(sgd1)
    c2 = c.set_params(penalty='l2')
    assert isinstance(c2, Chained)
    c3 = Chained(sgd2)
    assert c2._name == c3._name  # set_params name equivalent to init
    # Check no mutation
    assert c2.get_params()['penalty'] == 'l2'
    assert c2.compute().penalty == 'l2'
    assert c.get_params()['penalty'] == 'l1'
    assert c.compute().penalty == 'l1'


def test_setattr():
    c = Chained(sgd1)
    with pytest.raises(AttributeError):
        c.penalty = 'l2'


def test_getattr():
    c = Chained(sgd1)
    assert c.penalty == sgd1.penalty
    with pytest.raises(AttributeError):
        c.not_a_real_parameter


def test_dir():
    c = Chained(sgd1)
    attrs = dir(c)
    assert 'penalty' in attrs


def test_repr():
    c = Chained(sgd1)
    res = repr(c)
    assert res.startswith('Chained')


def fit_test(c, X, y, **kwargs):
    fit = c.fit(X, y, **kwargs)
    assert fit is not c
    assert isinstance(fit, Chained)

    res = fit.compute()
    assert isinstance(res, SGDClassifier)
    assert res.coef_ is not None
    assert fit.coef_ is None
    assert c.coef_ is None


def test_fit_dask_array():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit_test(c, X, y, classes=[0, 1, 2])


def test_fit_dask_matrix():
    X_bag = db.from_sequence([X_iris[0:20], X_iris[20:40], X_iris[40:]])
    y_bag = db.from_sequence([y_iris[0:20], y_iris[20:40], y_iris[40:]])
    X = dm.from_bag(X_bag)
    y = dm.from_bag(y_bag)

    c = Chained(sgd1)
    fit_test(c, X, y, classes=[0, 1, 2])


def test_predict():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit = c.fit(X, y, classes=[0, 1, 2])

    pred = fit.predict(X_iris)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    pred = fit.predict(X)
    assert isinstance(pred, da.Array)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    will_error = c.predict(X_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit = c.fit(X, y, classes=[0, 1, 2])

    s = fit.score(X_iris, y_iris)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    s = fit.score(X, y)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    will_error = c.score(X_iris, y_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()
