import os
from ast import literal_eval

import numpy as np
from distributed import Lock, Variable, get_worker
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import _num_samples, check_array


# This class doesn't inherit from BaseEstimator to test hyperparameter search
# on user-defined classifiers.
class MockClassifier:
    """Dummy classifier to test the parameter search algorithms"""

    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert len(X) == len(Y)
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        return T.shape[0]

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict
    inverse_transform = predict

    def transform(self, X):
        return X

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def get_params(self, deep=False):
        return {"foo_param": self.foo_param}

    def set_params(self, **params):
        self.foo_param = params["foo_param"]
        return self


class MockClassifierWithFitParam(MockClassifier):
    """A mock classifier with a required fit param."""

    def fit(self, X, y, mock_fit_param=None):
        if mock_fit_param is None:
            raise ValueError("Requires non-None 'mock_fit_param'")
        return super(MockClassifierWithFitParam, self)


class ScalingTransformer(BaseEstimator):
    def __init__(self, factor=1):
        self.factor = factor

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X * self.factor


class CheckXClassifier(BaseEstimator):
    """Used to check output of featureunions"""

    def __init__(self, expected_X=None):
        self.expected_X = expected_X

    def fit(self, X, y):
        assert (X == self.expected_X).all()
        assert len(X) == len(y)
        return self

    def predict(self, X):
        return X.sum(axis=1)

    def score(self, X=None, y=None):
        return self.predict(X)[0]


class FailingClassifier(BaseEstimator):
    """Classifier that raises a ValueError on fit()"""

    FAILING_PARAMETER = 2
    FAILING_SCORE_PARAMETER = object()
    FAILING_PREDICT_PARAMETER = object()

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y=None):
        if self.parameter == FailingClassifier.FAILING_PARAMETER:
            raise ValueError("Failing classifier failed as required")
        return self

    def transform(self, X):
        return X

    def predict(self, X):
        if self.parameter == self.FAILING_PREDICT_PARAMETER:
            raise ValueError("Failing during predict as required")
        return np.zeros(X.shape[0])

    def score(self, X, y):
        if self.parameter == self.FAILING_SCORE_PARAMETER:
            raise ValueError("Failing during score as required")
        return 0.5


# XXX: Mocking classes copied from sklearn.utils.mocking to remove nose
# dependency.  Can be removed when scikit-learn switches to pytest. See issue
# here: https://github.com/scikit-learn/scikit-learn/issues/7319


class ArraySlicingWrapper:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, aslice):
        return MockDataFrame(self.array[aslice])


class MockDataFrame:
    # have shape and length but don't support indexing.
    def __init__(self, array):
        self.array = array
        self.values = array
        self.shape = array.shape
        self.ndim = array.ndim
        # ugly hack to make iloc work.
        self.iloc = ArraySlicingWrapper(array)

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None):
        # Pandas data frames also are array-like: we want to make sure that
        # input validation in cross-validation does not try to call that
        # method.
        return self.array


class CheckingClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of X and y in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.
    """

    def __init__(
        self, check_y=None, check_X=None, foo_param=0, expected_fit_params=None
    ):
        self.check_y = check_y
        self.check_X = check_X
        self.foo_param = foo_param
        self.expected_fit_params = expected_fit_params

    def fit(self, X, y, **fit_params):
        assert len(X) == len(y)
        if self.check_X is not None:
            assert self.check_X(X)
        if self.check_y is not None:
            assert self.check_y(y)
        self.classes_ = np.unique(check_array(y, ensure_2d=False, allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            assert (
                len(missing) == 0
            ), "Expected fit parameter(s) %s not " "seen." % list(missing)
            for key, value in fit_params.items():
                assert len(value) == len(X), (
                    "Fit parameter %s has length"
                    "%d; expected %d." % (key, len(value), len(X),)
                )
        return self

    def predict(self, T):
        if self.check_X is not None:
            assert self.check_X(T)
        return self.classes_[np.zeros(_num_samples(T), dtype=np.int)]

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score


class AsCompletedEstimator(MockClassifier):
    def __init__(
        self, killed_workers_name, lock_name, counter_name, min_complete, foo_param=None
    ):
        super(AsCompletedEstimator, self).__init__(foo_param)
        self.counter_name = counter_name
        self.killed_workers_name = killed_workers_name
        self.lock_name = lock_name
        self.min_complete = min_complete

    def fit(self, X, y=None):
        w = get_worker()
        dsk_lock = Lock(self.lock_name, client=w.client)
        dsk_counter = Variable(self.counter_name, client=w.client)
        dsk_killed_workers = Variable(self.killed_workers_name, client=w.client)

        for e in list(w.executing):
            should_die = False
            t = literal_eval(e)
            with dsk_lock:
                c = dsk_counter.get()
                dsk_counter.set(c + 1)
                killed_workers = dsk_killed_workers.get()
                if c > self.min_complete and t not in killed_workers:
                    killed_workers[t] = True
                    should_die = True
                    dsk_killed_workers.set(killed_workers)

            if should_die:
                os.kill(os.getpid(), 9)
        return self


class LinearFunction(BaseEstimator):
    def __init__(self, intercept=0, slope=1, foo=0):
        self._num_calls = 0
        self.intercept = intercept
        self.slope = slope
        self.foo = foo
        super(LinearFunction, self).__init__()

    def fit(self, *args):
        return self

    def partial_fit(self, *args, **kwargs):
        self._num_calls += 1
        return self

    def score(self, *args, **kwargs):
        return self.intercept + self.slope * self._num_calls


class _MaybeLinearFunction(BaseEstimator):
    def __init__(self, final_score=1):
        self.final_score = final_score
        self._calls = 0

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        self._calls += 1

    def score(self, X, y):
        if self.final_score <= 3:
            return self.final_score * (1 - 1 / (self._calls + 2))
        return self.final_score
