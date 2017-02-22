import copy

import dask.array as da
from dask.base import Base, tokenize
from dask.delayed import delayed, Delayed

from sklearn.utils.validation import indexable


def _indexable(x):
    return indexable(x)[0]


def to_indexable(*args):
    """Ensure that all args are an indexable type.

    Conversion runs lazily for dask objects, immediately otherwise."""
    for x in args:
        if x is None or isinstance(x, da.Array):
            yield x
        elif isinstance(x, Base):
            yield delayed(_indexable, pure=True)(x)
        else:
            yield _indexable(x)


def to_keys(dsk, *args):
    for x in args:
        if x is None:
            yield None
        elif isinstance(x, da.Array):
            x = delayed(x)
            dsk.update(x.dask)
            yield x.key
        elif isinstance(x, Delayed):
            dsk.update(x.dask)
            yield x.key
        else:
            assert not isinstance(x, Base)
            key = 'array-' + tokenize(x)
            dsk[key] = x
            yield key


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)


def unzip(itbl, n):
    if itbl:
        return zip(*itbl)
    return [()] * n
