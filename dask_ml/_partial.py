from __future__ import absolute_import, division, print_function

import logging
import os

import dask
import numpy as np
import sklearn.utils
from dask.delayed import Delayed
from toolz import partial

logger = logging.getLogger(__name__)


def _partial_fit(model, x, y, kwargs=None):
    kwargs = kwargs or dict()
    model.partial_fit(x, y, **kwargs)
    return model


def fit(
    model,
    x,
    y,
    compute=True,
    shuffle_blocks=True,
    random_state=None,
    assume_equal_chunks=False,
    **kwargs
):
    """Fit scikit learn model against dask arrays

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

    nblocks, x_name = _blocks_and_name(x)
    if y is not None:
        y_nblocks, y_name = _blocks_and_name(y)
        assert y_nblocks == nblocks
    else:
        y_name = ""

    if not hasattr(model, "partial_fit"):
        msg = "The class '{}' does not implement 'partial_fit'."
        raise ValueError(msg.format(type(model)))

    order = list(range(nblocks))
    if shuffle_blocks:
        rng = sklearn.utils.check_random_state(random_state)
        rng.shuffle(order)

    name = "fit-" + dask.base.tokenize(model, x, y, kwargs, order)

    if hasattr(x, "chunks") and x.ndim > 1:
        x_extra = (0,)
    else:
        x_extra = ()

    dsk = {(name, -1): model}
    dsk.update(
        {
            (name, i): (
                _partial_fit,
                (name, i - 1),
                (x_name, order[i]) + x_extra,
                (y_name, order[i]),
                kwargs,
            )
            for i in range(nblocks)
        }
    )

    graphs = {x_name: x.__dask_graph__(), name: dsk}
    if hasattr(y, "__dask_graph__"):
        graphs[y_name] = y.__dask_graph__()

    try:
        from dask.highlevelgraph import HighLevelGraph

        new_dsk = HighLevelGraph.merge(*graphs.values())
    except ImportError:
        from dask import sharedict

        new_dsk = sharedict.merge(*graphs.values())

    value = Delayed((name, nblocks - 1), new_dsk)

    if compute:
        return value.compute()
    else:
        return value


def _blocks_and_name(obj):
    if hasattr(obj, "chunks"):
        nblocks = len(obj.chunks[0])
        name = obj.name

    elif hasattr(obj, "npartitions"):
        # dataframe, bag
        nblocks = obj.npartitions
        if hasattr(obj, "_name"):
            # dataframe
            name = obj._name
        else:
            # bag
            name = obj.name

    return nblocks, name


def _predict(model, x):
    return model.predict(x)[:, None]


def predict(model, x):
    """Predict with a scikit learn model

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
