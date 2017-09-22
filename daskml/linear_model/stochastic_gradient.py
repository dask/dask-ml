from sklearn.linear_model import stochastic_gradient as _sg
import numpy as np
from dask.array import learn
import dask
import dask.threaded


class _BigPartialFitMixin:

    def __init__(self, **kwargs):
        missing = set(self._init_kwargs) - set(kwargs)

        if missing:
            raise TypeError("{} requires the keyword arguments {}".format(
                type(self), missing)
            )
        for kwarg in self._init_kwargs:
            setattr(self, kwarg, kwargs.pop(kwarg))
        super().__init__(**kwargs)

    @classmethod
    def _get_param_names(cls):
        # Evil hack to make sure repr, get_params work
        # We could also try rewriting __init__ once the class is created
        bases = cls.mro()
        # walk bases until you hit an sklearn class.
        for base in bases:
            if base.__module__.startswith("sklearn"):
                break

        # merge the inits
        my_init = cls._init_kwargs
        their_init = base._get_param_names()
        return my_init + their_init

    def fit(self, X, y=None, get=None):
        if get is None:
            get = dask.threaded.get
        result = learn.fit(self, X, y, classes=self.classes, get=get)

        # Copy the learned attributes over to self
        # It should go without saying that this is *not* threadsafe
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)
        return self

    def predict(self, X, dtype=None):
        predict = super().predict
        if dtype is None:
            dtype = self._get_predict_dtype(X)
        return X.map_blocks(predict, dtype=dtype, drop_axis=1)

    def _get_predict_dtype(self, X):
        xx = np.zeros((1, X.shape[1]), dtype=X.dtype)
        return super().predict(xx).dtype


class BigSGDClassifier(_BigPartialFitMixin, _sg.SGDClassifier):

    _init_kwargs = ['classes']


class BigSGDRegressor(_BigPartialFitMixin, _sg.SGDRegressor):
    _init_kwargs = []
