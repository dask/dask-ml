import os
import six
from abc import ABCMeta

import numpy as np
import dask
from dask.array import learn


class _WritableDoc(ABCMeta):
    """In py27, classes inheriting from `object` do not have
    a multable __doc__.

    We inherit from ABCMeta instead of type to avoid metaclass
    conflicts, since some sklearn estimators (eventually) subclass
    ABCMeta
    """
    # TODO: Py2: remove all this


@six.add_metaclass(_WritableDoc)
class _BigPartialFitMixin(object):
    """ Wraps a partial_fit enabled estimator for use with Dask arrays """

    _init_kwargs = []
    _fit_kwargs = []

    def __init__(self, **kwargs):
        missing = set(self._init_kwargs) - set(kwargs)

        if missing:
            raise TypeError("{} requires the keyword arguments {}".format(
                type(self), missing)
            )
        for kwarg in self._init_kwargs:
            setattr(self, kwarg, kwargs.pop(kwarg))
        super(_BigPartialFitMixin, self).__init__(**kwargs)

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

        fit_kwargs = {k: getattr(self, k) for k in self._fit_kwargs}
        result = learn.fit(self, X, y, get=get, **fit_kwargs)

        # Copy the learned attributes over to self
        # It should go without saying that this is *not* threadsafe
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)
        return self

    def predict(self, X, dtype=None):
        predict = super(_BigPartialFitMixin, self).predict
        if dtype is None:
            dtype = self._get_predict_dtype(X)
        return X.map_blocks(predict, dtype=dtype, drop_axis=1)

    def _get_predict_dtype(self, X):
        xx = np.zeros((1, X.shape[1]), dtype=X.dtype)
        return super(_BigPartialFitMixin, self).predict(xx).dtype


def _copy_partial_doc(cls):
    for base in cls.mro():
        if base.__module__.startswith('sklearn'):
            break
    lines = base.__doc__.split(os.linesep)
    header, rest = lines[0], lines[1:]

    insert = """

    This class wraps scikit-learn's {classname}. When a dask-array is passed
    to our ``fit`` method, the array is passed block-wise to the scikit-learn
    class' ``partial_fit`` method. This will allow you to fit the estimator
    on larger-than memory datasets sequentially (block-wise), but without an
    parallelism, or any ability to distribute across a cluster.""".format(
        classname=base.__name__)

    doc = '\n'.join([header + insert] + rest)

    cls.__doc__ = doc
    return cls


__all__ = [
    '_BigPartialFitMixin',
    '_copy_partial_doc',
]
