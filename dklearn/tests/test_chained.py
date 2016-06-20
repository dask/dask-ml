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
from toolz import keymap, dissoc

import dklearn.matrix as dm
from dklearn.estimator import Estimator
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
    assert isinstance(c1.estimator, Estimator)

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
    assert tokenize(c1) != tokenize(c1.estimator)


def test_clone():
    c = Chained(sgd1)
    c2 = clone(c)
    assert (dissoc(c.get_params(), 'estimator') ==
            dissoc(c2.get_params(), 'estimator'))
    assert c._name == c2._name
    assert c.estimator is not c2.estimator
    assert c.estimator._est is not c2.estimator._est


def test__estimator_type():
    c = Chained(sgd1)
    assert c._estimator_type == sgd1._estimator_type


def test_get_params():
    c = Chained(sgd1)
    c_params = c.get_params()
    assert c_params['estimator'] == c.estimator
    del c_params['estimator']
    assert c_params == keymap(lambda k: 'estimator__' + k, sgd1.get_params())
    assert c.get_params(deep=False) == {'estimator': c.estimator}


def test_set_params():
    c = Chained(sgd1)
    c2 = c.set_params(estimator__penalty='l2')
    assert isinstance(c2, Chained)
    c3 = Chained(sgd2)
    assert c2._name == c3._name  # set_params name equivalent to init
    # Check no mutation
    assert c2.get_params()['estimator__penalty'] == 'l2'
    assert c2.compute().estimator.penalty == 'l2'
    assert c.get_params()['estimator__penalty'] == 'l1'
    assert c.compute().estimator.penalty == 'l1'

    # Changing estimator works
    c2 = c.set_params(estimator=sgd2)
    assert c2._name == c3._name
    assert isinstance(c2.estimator, Estimator)

    # Fast return
    c2 = c.set_params()
    assert c2 is c

    # ambiguous change
    with pytest.raises(ValueError):
        c.set_params(estimator=sgd2, estimator__penalty='l2')

    with pytest.raises(ValueError):
        c.set_params(not_a_real_param='foo')


def test_setattr():
    c = Chained(sgd1)
    with pytest.raises(AttributeError):
        c.estimator = sgd2


def fit_test(c, X, y):
    fit = c.fit(X, y)
    assert fit is not c
    assert fit.estimator is not c.estimator
    assert isinstance(fit, Chained)

    res = fit.compute()
    assert isinstance(res, Chained)
    assert isinstance(res.estimator, Estimator)
    assert res.estimator.coef_ is not None
    assert fit.estimator.coef_ is None
    assert c.estimator.coef_ is None


def test_fit_dask_array():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit_test(c, X, y)


def test_fit_dask_matrix():
    X_bag = db.from_sequence([X_iris[0:20], X_iris[20:40], X_iris[40:]])
    y_bag = db.from_sequence([y_iris[0:20], y_iris[20:40], y_iris[40:]])
    X = dm.from_bag(X_bag)
    y = dm.from_bag(y_bag)

    c = Chained(sgd1)
    fit_test(c, X, y)


def test_predict():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit = c.fit(X, y)

    pred = fit.predict(X_iris)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    pred = fit.predict(X)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    will_error = c.predict(X_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)

    c = Chained(sgd1)
    fit = c.fit(X, y)

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


def test_to_sklearn():
    c = Chained(sgd1)
    res = c.to_sklearn()
    assert isinstance(res, SGDClassifier)

    res = c.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), SGDClassifier)

    # After fitting
    X = da.from_array(X_iris, chunks=4)
    y = da.from_array(y_iris, chunks=4)
    fit = c.fit(X, y)
    res = fit.to_sklearn()
    assert isinstance(res, SGDClassifier)
    assert res.coef_ is not None
    assert fit.estimator.coef_ is None

    res = c.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), SGDClassifier)
    assert res.coef_ is not None
