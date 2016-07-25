from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import numpy.testing.utils as tm
import dask.array as da
import dask.bag as db
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import SGDClassifier, SGDRegressor

import dklearn.matrix as dm
from dklearn.wrapped import Wrapped
from dklearn.averaged import Averaged, merge_estimators

# Not fit estimators raise NotFittedError, but old versions of scikit-learn
# include two definitions of this error, which makes it hard to catch
# appropriately. Since it subclasses from `AttributeError`, this is good
# enough for the tests.
NotFittedError = AttributeError

sgdc_1 = SGDClassifier(penalty='l1')
sgdc_2 = SGDClassifier(penalty='l2')

sgdr = SGDRegressor(penalty='l1')

Xc, yc = make_classification(n_samples=1000, n_features=100, random_state=0)
Xr, yr = make_regression(n_samples=1000, n_features=100, random_state=0)


def test_multiclass_merge_estimators():
    coef1 = np.array([[1., 2, 3],
                      [4, 5, 6],
                      [7., 8, 9]])
    coef2 = np.array([[10., 11, 12]])
    coef3 = np.array([[13., 14, 15],
                      [16, 17, 18],
                      [19, 20, 21]])

    intercept1 = np.array([[1.], [2], [3]])
    intercept2 = np.array([[4.]])
    intercept3 = np.array([[5.], [6], [7]])

    classes1 = np.array([0, 1, 2])
    classes2 = np.array([1, 2])
    classes3 = np.array([1, 2, 3])

    def f(coef, intercept, classes):
        s = SGDClassifier()
        s.coef_ = coef
        s.intercept_ = intercept
        s.classes_ = classes
        return s

    ests = [f(coef1, intercept1, classes1),
            f(coef2, intercept2, classes2),
            f(coef3, intercept3, classes3)]

    coef = np.zeros((4, 3), dtype='f8')
    coef[classes1] += coef1
    coef[[classes2[1]]] += coef2
    coef[classes3] += coef3
    coef /= 3

    intercept = np.zeros((4, 1), dtype='f8')
    intercept[classes1] += intercept1
    intercept[[classes2[1]]] += intercept2
    intercept[classes3] += intercept3
    intercept /= 3

    res = merge_estimators(ests)
    tm.assert_almost_equal(res.coef_, coef)
    tm.assert_almost_equal(res.intercept_, intercept)
    tm.assert_almost_equal(res.classes_, np.arange(4))


def test_Averaged__init__():
    c1 = Averaged(sgdc_1)
    c2 = Averaged(sgdc_2)

    assert c1._name != c2._name
    assert Averaged(sgdc_1)._name == c1._name

    with pytest.raises(TypeError):
        Averaged("not an estimator")


def test_Averaged_from_sklearn():
    c1 = Averaged(sgdc_1)
    assert Averaged.from_sklearn(sgdc_1)._name == c1._name
    assert Averaged.from_sklearn(c1) is c1


def test_tokenize_Averaged():
    c1 = Averaged(sgdc_1)
    c2 = Averaged(sgdc_2)
    assert tokenize(c1) == tokenize(c1)
    assert tokenize(c1) != tokenize(c2)
    assert tokenize(c1) != tokenize(Wrapped(sgdc_1))


def test_clone():
    c = Averaged(sgdc_1)
    c2 = clone(c)
    assert isinstance(c2, Averaged)
    assert c._name == c2._name
    assert c._est is not c2._est


def test__estimator_type():
    c = Averaged(sgdc_1)
    assert c._estimator_type == sgdc_1._estimator_type


def test_get_params():
    c = Averaged(sgdc_1)
    assert c.get_params() == sgdc_1.get_params()
    assert c.get_params(deep=False) == sgdc_1.get_params(deep=False)


def test_set_params():
    c = Averaged(sgdc_1)
    c2 = c.set_params(penalty='l2')
    assert isinstance(c2, Averaged)
    c3 = Averaged(sgdc_2)
    assert c2._name == c3._name  # set_params name equivalent to init
    # Check no mutation
    assert c2.get_params()['penalty'] == 'l2'
    assert c2.compute().penalty == 'l2'
    assert c.get_params()['penalty'] == 'l1'
    assert c.compute().penalty == 'l1'


def test_setattr():
    c = Averaged(sgdc_1)
    with pytest.raises(AttributeError):
        c.penalty = 'l2'


def test_getattr():
    c = Averaged(sgdc_1)
    assert c.penalty == sgdc_1.penalty
    with pytest.raises(AttributeError):
        c.not_a_real_parameter


def test_dir():
    c = Averaged(sgdc_1)
    attrs = dir(c)
    assert 'penalty' in attrs


def test_repr():
    c = Averaged(sgdc_1)
    res = repr(c)
    assert res.startswith('Averaged')


def fit_test(c, X, y, cls):
    fit = c.fit(X, y)
    assert fit is not c
    assert isinstance(fit, Averaged)

    res = fit.compute()
    assert isinstance(res, cls)
    assert res.coef_ is not None
    assert fit.coef_ is None
    assert c.coef_ is None


def test_fit_dask_array():
    dXc = da.from_array(Xc, chunks=200)
    dyc = da.from_array(yc, chunks=200)

    c = Averaged(sgdc_1)
    fit_test(c, dXc, dyc, SGDClassifier)

    dXr = da.from_array(Xr, chunks=200)
    dyr = da.from_array(yr, chunks=200)

    c = Averaged(sgdr)
    fit_test(c, dXr, dyr, SGDRegressor)


def test_fit_dask_matrix():
    X_bag = db.from_sequence([Xc[0:200], Xc[200:400], Xc[400:]])
    y_bag = db.from_sequence([yc[0:200], yc[200:400], yc[400:]])
    dX = dm.from_bag(X_bag)
    dy = dm.from_bag(y_bag)

    c = Averaged(sgdc_1)
    fit_test(c, dX, dy, SGDClassifier)


def test_predict():
    dX = da.from_array(Xc, chunks=200)
    dy = da.from_array(yc, chunks=200)

    c = Averaged(sgdc_1)
    fit = c.fit(dX, dy)

    pred = fit.predict(Xc)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    pred = fit.predict(dX)
    assert isinstance(pred, da.Array)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    will_error = c.predict(Xc)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    dX = da.from_array(Xc, chunks=200)
    dy = da.from_array(yc, chunks=200)

    c = Averaged(sgdc_1)
    fit = c.fit(dX, dy)

    s = fit.score(Xc, yc)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    s = fit.score(dX, dy)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    will_error = c.score(Xc, yc)
    with pytest.raises(NotFittedError):
        will_error.compute()
