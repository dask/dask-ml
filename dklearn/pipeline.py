from __future__ import absolute_import, print_function, division

from dask.base import tokenize
from sklearn import pipeline

from .core import DaskBaseEstimator, from_sklearn


class Pipeline(DaskBaseEstimator, pipeline.Pipeline):
    _finalize = staticmethod(lambda res: pipeline.Pipeline(res[0]))

    def __init__(self, steps):
        # Run the sklearn init to validate the pipeline steps
        steps = pipeline.Pipeline(steps).steps
        steps = [(k, from_sklearn(v)) for k, v in steps]
        object.__setattr__(self, 'steps', steps)
        self._name = 'pipeline-' + tokenize(self.steps)

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

    @classmethod
    def from_sklearn(cls, est):
        if not isinstance(est, pipeline.Pipeline):
            raise TypeError("est must be a sklearn Pipeline")
        return cls(est.steps)

    def set_params(self, **params):
        if not params:
            return self
        if 'steps' in params:
            if len(params) == 1:
                return Pipeline(params['steps'])
            raise ValueError("Setting params with both `'steps'` and nested "
                             "parameters is ambiguous due to order of "
                             "operations. To change both steps and "
                             "sub-parameters create a new `Pipeline`.")
        # All params should be nested, or error
        sub_params = dict((n, {}) for n in self.named_steps)
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1 and split[0] in sub_params:
                # nested objects case
                sub_params[split[0]][split[1]] = value
            else:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self.__class__.__name__))
        steps = [(n, e.set_params(**sub_params[n])) for n, e in self.steps]
        return Pipeline(steps)

    def __setattr__(self, k, v):
        if k in ('_name', '_dask'):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Attribute setting not permitted. "
                                 "Use `set_params` to change parameters")

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
        fit, Xt = self._final_estimator._fit_transform(Xt, y, **params)
        fit_steps.append(fit)
        new_steps = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return Pipeline(new_steps), Xt


from_sklearn.dispatch.register(pipeline.Pipeline, Pipeline.from_sklearn)
