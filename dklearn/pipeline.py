from __future__ import absolute_import, print_function, division

from dask.base import tokenize
from sklearn import pipeline

from .core import DaskBaseEstimator, from_sklearn


class Pipeline(DaskBaseEstimator):

    def __init__(self, steps):
        self.steps = [(k, from_sklearn(v)) for k, v in steps]
        self._est = pipeline.Pipeline([(n, s._est) for n, s in self.steps])
        self._name = 'pipeline-' + tokenize('Pipeline', steps)

    @staticmethod
    def _finalize(res):
        return pipeline.Pipeline(res[0])

    @property
    def dask(self):
        if hasattr(self, '_dask'):
            return self._dask
        dsk = {}
        names = []
        tasks = []
        for n, s in self.steps:
            dsk.update(s.dask)
            names.append(n)
            tasks.append((s._finalize, s._keys()))
        dsk[self._name] = (list, (zip, names, tasks))
        self._dask = dsk
        return dsk

    def _keys(self):
        return [self._name]

    @classmethod
    def from_sklearn(cls, est):
        if not isinstance(est, pipeline.Pipeline):
            raise TypeError("est must be a sklearn Pipeline")
        return cls(est.steps)

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    @property
    def _estimator_type(self):
        return self._final_estimator._estimator_type

    def _pre_transform(self, Xt, y=None, **fit_params):
        # Separate out parameters
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        # Call fit_transform on all but last estimator
        fit_steps = []
        for name, transform in self.steps[:-1]:
            kwargs = fit_params_steps[name]
            fit, Xt = transform._fit_transform(Xt, y, **kwargs)
            fit_steps.append(fit)
        return fit_steps, Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        fit_steps, Xt, params = self._pre_transform(X, y, **fit_params)
        fit = self._final_estimator.fit(Xt, y, **params)
        fit_steps.append(fit)
        new_steps = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return Pipeline(new_steps)

    def transform(self, X):
        for name, transform in self.steps:
            X = transform.transform(X)
        return X

    def predict(self, X):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.predict(X)

    def score(self, X, y=None):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.score(X, y)

    def _fit_transform(self, X, y=None, **fit_params):
        fit_steps, Xt, params = self._pre_transform(X, y, **fit_params)
        fit, Xt = self._final_estimator._fit_transform(X, y, **params)
        fit_steps.append(fit)
        new_steps = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return Pipeline(new_steps), Xt


from_sklearn.dispatch.register(pipeline.Pipeline, Pipeline.from_sklearn)
