"""Wrappers for IID Learning"""
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import sklearn.base


class FirstBlockFitter(sklearn.base.BaseEstimator):
    """Meta-estimator for fitting on just the first block of
    a dask.array, or first partition of a dask.dataframe.

    Parameters
    ----------
    estimator : Estimator
        The underlying estimator that is fit.

    Notes
    -----
    The attributes learned by the underlying estimator are
    copied over to the 'FirstBlockFitter' after fitting.

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from dask_ml.datasets import make_classification
    >>> X, y = make_classification(n_samples=10000, chunks=1000)

    Wrap the regular classifier and fit on the first block (1000 samples).

    >>> clf = FirstBlockFitter(GradientBoostingClassifier())
    >>> clf.fit(X, y)
    FirstBlockFitter(estimator=GradientBoostingClassifier(...))

    Learned attributes are available

    >>> clf.classes_
    array([0, 1])

    Transform, predict are block-wise and return dask objects

    >>> clf.predict(X)
    dask.array<predict, shape=(10000,), dtype=int64, chunksize=(1000,)>

    Which can be computed in parallel

    >>> clf.predict_proba(X).compute()
    array([[0.99141094, 0.00858906],
           [0.93178389, 0.06821611],
           [0.99129105, 0.00870895],
           ...,
           [0.97996652, 0.02003348],
           [0.98087444, 0.01912556],
           [0.99407016, 0.00592984]])
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        """Fit the underlying estimator, using just the first block of data.

        Parameters
        ----------
        X, y : array or dataframe
            For dask arrays, the arrays must be chunked only along the rows.
            Just the first block of ``X`` and ``y`` are passed through to the
            underlying estimator's fit method

            For dask dataframes, just the first partition passed through.

            For other inputs, the entire object is passed through.

        \*\*kwargs : key-value pairs
            All remaining arguments are passed through to the underlying
            estimator's fit method.

        Returns
        -------
        self : object
        """
        X = _first_block(X)
        y = _first_block(y)
        result = self.estimator.fit(X, y, **kwargs)

        # Copy over learned attributes
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)

        return self

    def transform(self, X):
        """Transform block or partition-wise for dask inputs.

        If the underlying estimator does not have a ``transform`` method, then
        an ``AttributeError`` is raised.
        """
        transform = self._check_method('transform')

        if isinstance(X, da.Array):
            return X.map_blocks(transform)
        elif isinstance(X, dd._Frame):
            return _apply_partitionwise(X, transform)
        else:
            return transform(X)

    def score(self, X, y):
        # TODO: re-implement some scoring functions.
        return self.estimator.score(X, y)

    def predict(self, X):
        """Predict for X.

        For dask inputs, a dask array or dataframe is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        Parameters
        ----------
        X : array or dataframe

        Returns
        -------
        y : NumPy array, pandas DataFrame, or dask array or dataframe
        """
        predict = self._check_method('predict')

        if isinstance(X, da.Array):
            return X.map_blocks(predict, dtype='int', drop_axis=1)

        elif isinstance(X, dd._Frame):
            return _apply_partitionwise(X, predict)

        else:
            return predict(X)

    def predict_proba(self, X):
        """Predict for X.

        For dask inputs, a dask array or dataframe is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        If the underlying estimator does not have a ``predict_proba``
        method, then an ``AttributeError`` is raised.

        Parameters
        ----------
        X : array or dataframe

        Returns
        -------
        y : NumPy array, pandas DataFrame, or dask array or dataframe
        """

        predict_proba = self._check_method('predict_proba')

        if isinstance(X, da.Array):
            # XXX: multiclass
            return X.map_blocks(predict_proba,
                                dtype='float',
                                chunks=(X.chunks[0], len(self.classes_)))
        elif isinstance(X, dd._Frame):
            return _apply_partitionwise(X, predict_proba)
        else:
            return predict_proba(X)

    def _check_method(self, method):
        """Check if self.estimator has 'method'.

        Raises
        ------
        AttributeError
        """
        if not hasattr(self.estimator, method):
            msg = ("The wrapped estimator '{}' does not have a "
                   "'{}' method.".format(self.estimator, method))
            raise AttributeError(msg)
        return getattr(self.estimator, method)


def _first_block(dask_object):
    """Extract the first block / partition from a dask object
    """
    if isinstance(dask_object, da.Array):
        if dask_object.ndim > 1 and dask_object.numblocks[-1] != 1:
            raise NotImplementedError("IID estimators require that the array "
                                      "blocked only along the first axis. "
                                      "Rechunk your array before fitting.")
        shape = (dask_object.chunks[0][0],)
        if dask_object.ndim > 1:
            shape = shape + (dask_object.chunks[1][0],)

        return da.from_delayed(dask_object.to_delayed().flatten()[0],
                               shape,
                               dask_object.dtype)

    if isinstance(dask_object, dd._Frame):
        return dask_object.get_partition(0)

    else:
        return dask_object


def _apply_partitionwise(X, func):
    """Apply a prediction partition-wise to a dask.dataframe"""
    sample = func(X._meta_nonempty)
    if sample.ndim <= 1:
        p = ()
    else:
        p = (sample.shape[1],)

    if isinstance(sample, np.ndarray):
        blocks = X.to_delayed()
        arrays = [
            da.from_delayed(dask.delayed(func)(block),
                            shape=(np.nan,) + p,
                            dtype=sample.dtype)
            for block in blocks
        ]
        return da.concatenate(arrays)
    else:
        return X.map_partitions(func, meta=sample)
