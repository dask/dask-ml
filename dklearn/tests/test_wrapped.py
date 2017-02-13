from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from dklearn import from_sklearn
from dklearn.wrapped import Wrapped

# Not fit estimators raise NotFittedError, but old versions of scikit-learn
# include two definitions of this error, which makes it hard to catch
# appropriately. Since it subclasses from `AttributeError`, this is good
# enough for the tests.
NotFittedError = AttributeError

clf1 = LogisticRegression(C=1000)
clf2 = LogisticRegression(C=5000)

iris = load_iris()
X_iris = iris.data[:, :2]
y_iris = iris.target


def test_tokenize_BaseEstimator():
    assert tokenize(clf1) == tokenize(clf1)
    assert tokenize(clf1) == tokenize(clone(clf1))
    assert tokenize(clf1) != tokenize(clf2)
    fit = clone(clf1).fit(X_iris, y_iris)
    assert tokenize(fit) == tokenize(fit)
    assert tokenize(fit) != tokenize(clf1)
    fit2 = clone(clf2).fit(X_iris, y_iris)
    assert tokenize(fit) != tokenize(fit2)


def test_from_sklearn():
    d = from_sklearn(clf1)
    assert from_sklearn(clf1)._name == d._name
    assert from_sklearn(clf2)._name != d._name
    assert 'logisticregression' in d._name.lower()


def test_Estimator__init__():
    d = Wrapped(LogisticRegression(C=1000))
    assert d._name == from_sklearn(clf1)._name

    with pytest.raises(ValueError):
        Wrapped(clf1, name='foo')

    with pytest.raises(TypeError):
        Wrapped("not an estimator")


def test_clone():
    d = Wrapped(clf1)
    d2 = clone(d)
    assert d.get_params() == d2.get_params()
    assert d._est is not d2._est


def test__estimator_type():
    d = Wrapped(clf1)
    assert d._estimator_type == clf1._estimator_type


def test_get_params():
    d = from_sklearn(clf1)
    assert d.get_params() == clf1.get_params()
    assert d.get_params(deep=False) == clf1.get_params(deep=False)


def test_set_params():
    d = from_sklearn(clf1)
    d2 = d.set_params(C=5)
    assert isinstance(d2, Wrapped)
    # Check no mutation
    assert d2.get_params()['C'] == 5
    assert d2.compute().C == 5
    assert d.get_params()['C'] == 1000
    assert d.compute().C == 1000


def test_setattr():
    d = Wrapped(clf1)
    with pytest.raises(AttributeError):
        d.C = 10


def test_getattr():
    d = Wrapped(clf1)
    assert d.C == clf1.C
    with pytest.raises(AttributeError):
        d.not_a_real_parameter


def test_dir():
    d = Wrapped(clf1)
    attrs = dir(d)
    assert 'C' in attrs


def test_repr():
    d = Wrapped(clf1)
    res = repr(d)
    assert res.startswith('Wrapped')


def test_fit():
    d = from_sklearn(clf1)
    fit = d.fit(X_iris, y_iris)
    assert fit is not d
    assert isinstance(fit, Wrapped)
    assert any('logisticregression' in k.lower() for k in fit.dask)

    res = fit.compute()
    assert hasattr(res, 'coef_')
    assert not hasattr(clf1, 'coef_')
    assert isinstance(res, LogisticRegression)


def test_predict():
    d = from_sklearn(clf1)
    fit = d.fit(X_iris, y_iris)
    pred = fit.predict(X_iris)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)
    assert any('logisticregression' in k.lower() for k in pred.dask)

    will_error = d.predict(X_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    d = from_sklearn(clf1)
    fit = d.fit(X_iris, y_iris)
    s = fit.score(X_iris, y_iris)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    dX_iris = da.from_array(X_iris, chunks=4)
    dy_iris = da.from_array(y_iris, chunks=4)
    s = fit.score(dX_iris, dy_iris)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    will_error = d.score(X_iris, y_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()
