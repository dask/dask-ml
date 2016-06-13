from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn import clone, pipeline
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from dklearn import from_sklearn
from dklearn.estimator import Estimator
from dklearn.pipeline import Pipeline

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

pipe1 = pipeline.Pipeline(steps=[('pca', PCA()),
                                 ('logistic', LogisticRegression(C=1000))])

pipe2 = clone(pipe1).set_params(pca__n_components=20, logistic__C=100)


def test_tokenize_sklearn_pipeline():
    assert tokenize(pipe1) == tokenize(pipe1)
    assert tokenize(pipe1) == tokenize(clone(pipe1))
    assert tokenize(pipe1) != tokenize(pipe2)
    fit = clone(pipe1).fit(X_digits, y_digits)
    assert tokenize(fit) == tokenize(fit)
    assert tokenize(fit) != tokenize(pipe1)
    fit2 = clone(pipe2).fit(X_digits, y_digits)
    assert tokenize(fit) != tokenize(fit2)


def test_pipeline():
    d = from_sklearn(pipe1)
    assert isinstance(d, Pipeline)
    assert from_sklearn(pipe1)._name == d._name
    assert from_sklearn(pipe2)._name != d._name

    # dask graph is cached on attribute access
    assert d.dask is d.dask


def test_get_params():
    d = from_sklearn(pipe1)
    assert d.get_params() == pipe1.get_params()
    assert d.get_params(deep=False) == pipe1.get_params(deep=False)


def test_set_params():
    d = from_sklearn(pipe1)
    d2 = d.set_params(pca__n_components=20, logistic__C=100)
    assert isinstance(d2, Pipeline)
    assert d2._name == from_sklearn(pipe2)._name
    # Check no mutation
    assert d2.get_params()['logistic__C'] == 100
    assert d2.compute().get_params()['logistic__C'] == 100
    assert d.get_params()['logistic__C'] == 1000
    assert d.compute().get_params()['logistic__C'] == 1000


def test_named_steps():
    d = from_sklearn(pipe1)
    steps = d.named_steps
    assert isinstance(steps['pca'], Estimator)
    assert isinstance(steps['logistic'], Estimator)


def test_fit():
    d = from_sklearn(pipe1)
    fit = d.fit(X_digits, y_digits)
    assert fit is not d
    assert isinstance(fit, Pipeline)

    res = fit.compute()
    assert isinstance(res, pipeline.Pipeline)
    assert hasattr(res, 'classes_')
    assert not hasattr(pipe1, 'classes_')


def test_predict():
    d = from_sklearn(pipe1)
    fit = d.fit(X_digits, y_digits)
    pred = fit.predict(X_digits)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)
    will_error = d.predict(X_digits)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    d = from_sklearn(pipe1)
    fit = d.fit(X_digits, y_digits)
    s = fit.score(X_digits, y_digits)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)
    will_error = d.score(X_digits, y_digits)
    with pytest.raises(NotFittedError):
        will_error.compute()
