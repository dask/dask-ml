from __future__ import absolute_import, division, print_function

import logging
import os
import warnings
from abc import ABCMeta

import dask
import numpy as np
import six
import sklearn.utils
from dask.delayed import Delayed
from toolz import partial

from ._utils import copy_learned_attributes

logger = logging.getLogger(__name__)


class _WritableDoc(ABCMeta):
    """In py27, classes inheriting from `object` do not have
    a multable __doc__.

    We inherit from ABCMeta instead of type to avoid metaclass
    conflicts, since some sklearn estimators (eventually) subclass
    ABCMeta
    """

    # TODO: Py2: remove all this


_partial_deprecation = (
    "'{cls.__name__}' is deprecated. Use "
    "'dask_ml.wrappers.Incremental({base.__name__}(), **kwargs)' "
    "instead."
)


@six.add_metaclass(_WritableDoc)
class _BigPartialFitMixin(object):
    """ Wraps a partial_fit enabled estimator for use with Dask arrays """

    _init_kwargs = []
    _fit_kwargs = []

    def __init__(self, **kwargs):
        self._deprecated()
        missing = set(self._init_kwargs) - set(kwargs)

        if missing:
            raise TypeError(
                "{} requires the keyword arguments {}".format(type(self), missing)
            )
        for kwarg in self._init_kwargs:
            setattr(self, kwarg, kwargs.pop(kwarg))
        super(_BigPartialFitMixin, self).__init__(**kwargs)

    @classmethod
    def _deprecated(cls):
        for base in cls.mro():
            if base.__module__.startswith("sklearn"):
                break

        warnings.warn(_partial_deprecation.format(cls=cls, base=base), FutureWarning)

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

    def fit(self, X, y=None, compute=True):
        fit_kwargs = {k: getattr(self, k) for k in self._fit_kwargs}
        result = fit(self, X, y, compute=compute, **fit_kwargs)

        if compute:
            copy_learned_attributes(result, self)
            return self
        return result

    def predict(self, X, dtype=None):
        predict = super(_BigPartialFitMixin, self).predict
        if dtype is None:
            dtype = self._get_predict_dtype(X)
        if isinstance(X, np.ndarray):
            return predict(X)
        return X.map_blocks(predict, dtype=dtype, drop_axis=1)

    def _get_predict_dtype(self, X):
        xx = np.zeros((1, X.shape[1]), dtype=X.dtype)
        return super(_BigPartialFitMixin, self).predict(xx).dtype


def _partial_fit(model, x, y, kwargs=None):
    kwargs = kwargs or dict()
    model.partial_fit(x, y, **kwargs)
    return model


def fit(model, x, y, compute=True, shuffle_blocks=True, random_state=None, **kwargs):
    """ Fit scikit learn model against dask arrays

    Model must support the ``partial_fit`` interface for online or batch
    learning.

    Ideally your rows are independent and identically distributed. By default,
    this function will step through chunks of the arrays in random order.

    Parameters
    ----------
    model: sklearn model
        Any model supporting partial_fit interface
    x: dask Array
        Two dimensional array, likely tall and skinny
    y: dask Array
        One dimensional array with same chunks as x's rows
    compute : bool
        Whether to compute this result
    shuffle_blocks : bool
        Whether to shuffle the blocks with ``random_state`` or not
    random_state : int or numpy.random.RandomState
        Random state to use when shuffling blocks
    kwargs:
        options to pass to partial_fit

    Examples
    --------
    >>> import dask.array as da
    >>> X = da.random.random((10, 3), chunks=(5, 3))
    >>> y = da.random.randint(0, 2, 10, chunks=(5,))

    >>> from sklearn.linear_model import SGDClassifier
    >>> sgd = SGDClassifier()

    >>> sgd = da.learn.fit(sgd, X, y, classes=[1, 0])
    >>> sgd  # doctest: +SKIP
    SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
           fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
           loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
           random_state=None, shuffle=False, verbose=0, warm_start=False)

    This passes all of X and y through the classifier sequentially.  We can use
    the classifier as normal on in-memory data

    >>> import numpy as np
    >>> sgd.predict(np.random.random((4, 3)))  # doctest: +SKIP
    array([1, 0, 0, 1])

    Or predict on a larger dataset

    >>> z = da.random.random((400, 3), chunks=(100, 3))
    >>> da.learn.predict(sgd, z)  # doctest: +SKIP
    dask.array<x_11, shape=(400,), chunks=((100, 100, 100, 100),), dtype=int64>
    """
    if not hasattr(x, "chunks") and hasattr(x, "to_dask_array"):
        x = x.to_dask_array()
    assert x.ndim == 2
    if y is not None:
        if not hasattr(y, "chunks") and hasattr(y, "to_dask_array"):
            y = y.to_dask_array()
        assert y.ndim == 1
        assert x.chunks[0] == y.chunks[0]
    assert hasattr(model, "partial_fit")
    if len(x.chunks[1]) > 1:
        x = x.rechunk(chunks=(x.chunks[0], sum(x.chunks[1])))

    nblocks = len(x.chunks[0])
    order = list(range(nblocks))
    if shuffle_blocks:
        rng = sklearn.utils.check_random_state(random_state)
        rng.shuffle(order)

    name = "fit-" + dask.base.tokenize(model, x, y, kwargs, order)
    dsk = {(name, -1): model}
    dsk.update(
        {
            (name, i): (
                _partial_fit,
                (name, i - 1),
                (x.name, order[i], 0),
                (getattr(y, "name", ""), order[i]),
                kwargs,
            )
            for i in range(nblocks)
        }
    )

    new_dsk = dask.sharedict.merge((name, dsk), x.dask, getattr(y, "dask", {}))
    value = Delayed((name, nblocks - 1), new_dsk)

    if compute:
        return value.compute()
    else:
        return value


def _predict(model, x):
    return model.predict(x)[:, None]


def predict(model, x):
    """ Predict with a scikit learn model

    Parameters
    ----------
    model : scikit learn classifier
    x : dask Array

    See docstring for ``da.learn.fit``
    """
    if not hasattr(x, "chunks") and hasattr(x, "to_dask_array"):
        x = x.to_dask_array()
    assert x.ndim == 2
    if len(x.chunks[1]) > 1:
        x = x.rechunk(chunks=(x.chunks[0], sum(x.chunks[1])))
    func = partial(_predict, model)
    xx = np.zeros((1, x.shape[1]), dtype=x.dtype)
    dt = model.predict(xx).dtype
    return x.map_blocks(func, chunks=(x.chunks[0], (1,)), dtype=dt).squeeze()


def _copy_partial_doc(cls):
    for base in cls.mro():
        if base.__module__.startswith("sklearn"):
            break
    lines = base.__doc__.split(os.linesep)
    header, rest = lines[0], lines[1:]

    insert = """

    .. deprecated:: 0.6.0
       Use the :class:`dask_ml.wrappers.Incremental` meta-estimator instead.

    This class wraps scikit-learn's {classname}. When a dask-array is passed
    to our ``fit`` method, the array is passed block-wise to the scikit-learn
    class' ``partial_fit`` method. This will allow you to fit the estimator
    on larger-than memory datasets sequentially (block-wise), but without an
    parallelism, or any ability to distribute across a cluster.""".format(
        classname=base.__name__
    )

    doc = "\n".join([header + insert] + rest)

    cls.__doc__ = doc
    return cls
