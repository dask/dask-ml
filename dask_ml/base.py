import inspect
import os
from abc import ABCMeta

import dask
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import six
import sklearn.base
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
    """Insert a qualification into the base class' docstring."""
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


def _iid_rewriter(cls):
    """Insert a qualification into the base class' method's docstring.

    This rewrites for

    * fit
    * transform
    * predict
    * predict_proba
    """
    for base in cls.mro():
        if base.__module__.startswith('sklearn'):
            break

    fit_tpl = """

    *This class trains only on the first block of a dask array
    or the first partition of a dask dataframe.*"""

    other_tpl = """

    *This class operates block-wise on dask arrays and partition-wise on
    dask dataframes.*"""

    for name in ['fit', 'transform', 'predict', 'predict_proba']:
        method = getattr(base, name, None)
        if method and method.__doc__:
            lines = method.__doc__.split(os.linesep)
            header, rest = lines[0], lines[1:]

            if name == 'fit':
                insert = fit_tpl
            else:
                insert = other_tpl

            doc = '\n'.join([header + insert] + rest)
            setattr(getattr(base, name), '__doc__', doc)

    return cls


class _IIDBaseMixin:
    # Make a scikit-learn class transparently "work" with dask arrays.
    # We fit on just the first block / partition.
    # We transform, predict, etc. block- / partition-wise
    # Transformer, predict, etc. are implemented elsewhere, since not all
    # classes implement these methods.
    def fit(self, X, y=None):
        X = _first_block(X)
        y = _first_block(y)
        X, y = dask.compute(X, y)
        result = super().fit(X, y)

        # Copy the learned attributes over to self
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)

        return self


class _PredictMixin:
    def predict(self, X):
        predict = super().predict

        if isinstance(X, da.Array):
            return X.map_blocks(predict, dtype='int', drop_axis=1)
        elif isinstance(X, dd._Frame):
            sample = super().predict(X._meta_nonempty)
            if sample.ndim <= 1:
                p = ()
            elif sample.ndim == 1:
                p = sample.shape[1]
            else:
                raise AssertionError

            if isinstance(sample, np.ndarray):
                blocks = X.to_delayed()
                arrays = [
                    da.from_delayed(dask.delayed(predict)(block),
                                    shape=(np.nan,) + p,
                                    dtype=sample.dtype)
                    for block in blocks
                ]
                return da.concatenate(arrays)
            else:
                return X.map_partitions(super().predict, meta=sample)
        else:
            return super().predict(X)


class _PredictProbaMixin:
    def predict_proba(self, X):
        if isinstance(X, da.Array):
            # TODO: multiclass
            return X.map_blocks(super().predict_proba, dtype='float',
                                chunks=(X.chunks[0], 2))
        elif isinstance(X, dd._Frame):
            raise NotImplementedError("Not implemented yet")
        else:
            return super().predict_proba(X)


class _IIDTransformerMixin:
    def transform(self, X, y=None):
        if isinstance(X, da.Array):
            return X.map_blocks(super().transform, y)
        elif isinstance(X, dd._Frame):
            return X.map_partitions(super().transform, y)
        else:
            return super().transform(X, y)


def _first_block(dask_object):
    """Extract the first block / partition from a dask object
    """
    if isinstance(dask_object, da.Array):
        if dask_object.ndim > 1 and dask_object.numblocks[-1] != 1:
            raise NotImplementedError("IID estimators require that the array "
                                      "blocked only along the first axis. "
                                      "Rechunk your array before fitting.")
        return dask_object.to_delayed().flatten()[0]

    if isinstance(dask_object, dd._Frame):
        return dask_object.get_partition(0)


def _make_estimator(parent):
    bases = [_IIDBaseMixin]
    if issubclass(parent, sklearn.base.TransformerMixin):
        bases.append(_IIDTransformerMixin)

    d = {}
    if hasattr(parent, 'predict'):
        bases.append(_PredictMixin)
    if hasattr(parent, 'predict_proba'):
        bases.append(_PredictProbaMixin)

    bases.append(parent)

    # create the class from our component base classes.
    cls = type(parent.__name__, tuple(bases), d)
    # Add our qualifications to the docstirngs
    cls = _iid_rewriter(cls)
    # for pickle
    cls.__module__ = 'dask_ml.iid.' + parent.__module__.split('.')[1]
    return cls


def _find_estimators(module):
    exclusions = ()

    return [module.__dict__[cls] for cls in module.__all__
            if inspect.isclass(module.__dict__[cls]) and
            issubclass(module.__dict__[cls],
                       sklearn.base.BaseEstimator) and
            not issubclass(module.__dict__[cls], exclusions)]


__all__ = [
    '_BigPartialFitMixin',
    '_copy_partial_doc',
    '_make_estimator',
    '_find_estimators',
]
