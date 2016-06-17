from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn import pipeline
from sklearn.base import clone, BaseEstimator
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from toolz import dissoc

from dklearn import from_sklearn
from dklearn.estimator import Estimator
from dklearn.pipeline import Pipeline

digits = load_digits()
X_digits = digits.data
y_digits = digits.target

steps = [('pca', PCA()), ('logistic', LogisticRegression(C=1000))]

pipe1 = pipeline.Pipeline(steps=steps)
pipe2 = clone(pipe1).set_params(pca__n_components=20, logistic__C=100)


# Not fit estimators raise NotFittedError, but old versions of scikit-learn
# include two definitions of this error, which makes it hard to catch
# appropriately. Since it subclasses from `AttributeError`, this is good
# enough for the tests.
NotFittedError = AttributeError


class MissingMethods(BaseEstimator):
    """Small class to test pipeline constructor."""
    pass


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

    with pytest.raises(TypeError):
        Pipeline.from_sklearn("not an estimator")


def test_Pipeline__init__():
    d = Pipeline(steps)
    assert d._name == from_sklearn(pipe1)._name

    with pytest.raises(TypeError):
        Pipeline([MissingMethods(), LogisticRegression()])

    with pytest.raises(TypeError):
        Pipeline([PCA(), MissingMethods()])


def test_clone():
    d = Pipeline(steps)
    d2 = clone(d)
    assert d is not d2
    assert d._name == d2._name
    lr1 = d.named_steps['logistic']
    lr2 = d2.named_steps['logistic']
    assert lr1 is not lr2
    assert lr1.get_params() == lr2.get_params()
    assert isinstance(lr2, Estimator)


def test__estimator_type():
    d = Pipeline(steps)
    assert d._estimator_type == pipe1._estimator_type


def test_get_params():
    d = from_sklearn(pipe1)
    params1 = d.get_params()
    params2 = pipe1.get_params()
    assert (dissoc(params1, 'steps', 'logistic', 'pca') ==
            dissoc(params2, 'steps', 'logistic', 'pca'))
    params1 = d.get_params(deep=False)
    params2 = pipe1.get_params(deep=False)
    for dkstep, skstep in zip(params1['steps'], params2['steps']):
        # names are equivalent
        assert dkstep[0] == skstep[0]
        # ests have same params
        assert dkstep[1].get_params() == skstep[1].get_params()


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

    # Changing steps with set_params works
    d2 = d.set_params(steps=steps)
    assert d2._name == d._name
    lr1 = d.named_steps['logistic']
    lr2 = d2.named_steps['logistic']
    assert lr1.get_params() == lr2.get_params()
    assert isinstance(lr2, Estimator)

    # Fast return
    d2 = d.set_params()
    assert d2 is d

    # ambiguous change
    with pytest.raises(ValueError):
        d.set_params(steps=steps, logistic__C=10)

    with pytest.raises(ValueError):
        d.set_params(not_a_real_param='foo')


def test_named_steps():
    d = from_sklearn(pipe1)
    steps = d.named_steps
    assert isinstance(steps['pca'], Estimator)
    assert isinstance(steps['logistic'], Estimator)


def test_setattr():
    d = from_sklearn(pipe1)
    with pytest.raises(AttributeError):
        d.C = 10


def test_fit():
    d = from_sklearn(pipe1)
    fit = d.fit(X_digits, y_digits)
    assert fit is not d
    assert isinstance(fit, Pipeline)

    res = fit.compute()
    assert isinstance(res, Pipeline)
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


def test_to_sklearn():
    d = from_sklearn(pipe1)
    res = d.to_sklearn()
    assert isinstance(res, pipeline.Pipeline)
    assert isinstance(res.named_steps['logistic'], LogisticRegression)

    res = d.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), pipeline.Pipeline)

    # After fitting
    fit = d.fit(X_digits, y_digits)
    res = fit.to_sklearn()
    assert isinstance(res, pipeline.Pipeline)
    assert isinstance(res.named_steps['logistic'], LogisticRegression)

    res = d.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), pipeline.Pipeline)
