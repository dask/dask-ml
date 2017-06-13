from functools import wraps
from sklearn import datasets as _datasets
import dask.array as da


__all__ = []


def _wrap_maker(func):
    @wraps(func)
    def inner(*args, **kwargs):
        chunks = kwargs.pop('chunks')
        X, y = func(*args, **kwargs)
        return (da.from_array(X, chunks=(chunks, X.shape[-1])),
                da.from_array(y, chunks=chunks))
    __all__.append(func.__name__)
    return inner


make_classification = _wrap_maker(_datasets.make_classification)
make_regression = _wrap_maker(_datasets.make_regression)
