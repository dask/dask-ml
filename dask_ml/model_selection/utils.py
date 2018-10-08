import copy
import warnings
from distutils.version import LooseVersion

import dask
import dask.array as da
import scipy.sparse as sp
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from sklearn.utils import safe_indexing
from sklearn.utils.validation import _is_arraylike, indexable

from ..utils import _num_samples

if LooseVersion(dask.__version__) > "0.15.4":
    from dask.base import is_dask_collection
else:
    from dask.base import Base

    def is_dask_collection(x):
        return isinstance(x, Base)


def _indexable(x):
    return indexable(x)[0]


def _maybe_indexable(x):
    return indexable(x)[0] if _is_arraylike(x) else x


def to_indexable(*args, **kwargs):
    """Ensure that all args are an indexable type.

    Conversion runs lazily for dask objects, immediately otherwise.

    Parameters
    ----------
    args : array_like or scalar
    allow_scalars : bool, optional
        Whether to allow scalars in args. Default is False.
    """
    if kwargs.get("allow_scalars", False):
        indexable = _maybe_indexable
    else:
        indexable = _indexable
    for x in args:
        if x is None or isinstance(x, da.Array):
            yield x
        elif is_dask_collection(x):
            yield delayed(indexable, pure=True)(x)
        else:
            yield indexable(x)


def _index_param_value(num_samples, v, indices):
    """Private helper function for parameter value indexing.

    This determines whether a fit parameter `v` to a SearchCV.fit
    should be indexed along with `X` and `y`. Note that this differs
    from the scikit-learn version. They pass `X` and compute num_samples.
    We pass `num_samples` instead.
    """
    if not _is_arraylike(v) or _num_samples(v) != num_samples:
        # pass through: skip indexing
        return v
    if sp.issparse(v):
        v = v.tocsr()
    return safe_indexing(v, indices)


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
            assert not is_dask_collection(x)
            key = "array-" + tokenize(x)
            dsk[key] = x
            yield key


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)


def unzip(itbl, n):
    return zip(*itbl) if itbl else [()] * n


class DeprecationDict(dict):
    """A dict which raises a warning when some keys are looked up
    Note, this does not raise a warning for __contains__ and iteration.
    It also will raise a warning even after the key has been manually set by
    the user.

    This implementation was copied from Scikit-Learn.

    See License information here:
    https://github.com/scikit-learn/scikit-learn/blob/master/README.rst
    """

    def __init__(self, *args, **kwargs):
        self._deprecations = {}
        super(DeprecationDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key in self._deprecations:
            warn_args, warn_kwargs = self._deprecations[key]
            warnings.warn(*warn_args, **warn_kwargs)
        return super(DeprecationDict, self).__getitem__(key)

    def get(self, key, default=None):
        """Return the value corresponding to key, else default.

        Parameters
        ----------
        key : any hashable object
            The key
        default : object, optional
            The default returned when key is not in dict
        """
        # dict does not implement it like this, hence it needs to be overridden
        try:
            return self[key]
        except KeyError:
            return default

    def add_warning(self, key, *args, **kwargs):
        """Add a warning to be triggered when the specified key is read

        Parameters
        ----------
        key : any hashable object
            The key
        """
        self._deprecations[key] = (args, kwargs)
