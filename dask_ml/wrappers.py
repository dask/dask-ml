"""Meta-estimators for parallelizing scikit-learn."""
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import sklearn.base


class ParallelPostFit(sklearn.base.BaseEstimator):
    """Meta-estimator for parallel predict and transform.

    Parameters
    ----------
    estimator : Estimator
        The underlying estimator that is fit.

    Notes
    -----

    .. warning::

       This class is not appropriate for parallel or distributed *training*
       on large datasets.

    This estimator does not parallelize the training step. This simply calls
    the underlying estimators's ``fit`` method called and copies over the
    learned attributes to ``self`` afterwards.

    It is helpful for situations where your training dataset is relatively
    small (fits on a single machine) but you need to predict or transform
    a much larger dataset. ``predict``, ``predict_proba`` and ``transform``
    will be done in parallel (potentially distributed if you've connected
    to a ``dask.distributed.Client``).

    Note that many scikit-learn estimators already predict and transform in
    parallel. This meta-estimator may still be useful in those cases when your
    dataset is larger than memory, as the distributed scheduler will ensure the
    data isn't all read into memory at once.

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> import sklearn.datasets
    >>> import dask_ml.datasets

    Make a small 1,000 sample 2 training dataset and fit normally.

    >>> X, y = sklearn.datasets.make_classification(n_samples=1000,
    ...                                             random_state=0)
    >>> clf = ParallelPostFit(estimator=GradientBoostingClassifier())
    >>> clf.fit(X, y)
    ParallelPostFit(estimator=GradientBoostingClassifier(...))

    >>> clf.classes_
    array([0, 1])

    Transform and predict return dask outputs for dask inputs.

    >>> X_big, y_big = dask_ml.datasets.make_classification(n_samples=100000,
                                                            random_state=0)

    >>> clf.predict(X)
    dask.array<predict, shape=(10000,), dtype=int64, chunksize=(1000,)>

    Which can be computed in parallel.

    >>> clf.predict_proba(X).compute()
    array([[0.99141094, 0.00858906],
           [0.93178389, 0.06821611],
           [0.99129105, 0.00870895],
           ...,
           [0.97996652, 0.02003348],
           [0.98087444, 0.01912556],
           [0.99407016, 0.00592984]])
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        """Fit the underlying estimator.

        Parameters
        ----------
        X, y : array-like

        Returns
        -------
        self : object
        """
        result = self.estimator.fit(X, y, **kwargs)

        # Copy over learned attributes
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)

        return self

    def transform(self, X):
        """Transform block or partition-wise for dask inputs.

        For dask inputs, a dask array or dataframe is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        If the underlying estimator does not have a ``transform`` method, then
        an ``AttributeError`` is raised.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        transformed : array-like
        """
        transform = self._check_method('transform')

        if isinstance(X, da.Array):
            return X.map_blocks(transform)
        elif isinstance(X, dd._Frame):
            return _apply_partitionwise(X, transform)
        else:
            return transform(X)

    def score(self, X, y):
        """Returns the score on the given data.

        This uses the scoring defined by ``estimator.score``. This is
        currently immediate and sequential. In the future, this will be
        delayed and parallel.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
                return self.estimator.score(X, y)
        """
        return self.estimator.score(X, y)

    def predict(self, X):
        """Predict for X.

        For dask inputs, a dask array or dataframe is returned. For other
        inputs (NumPy array, pandas dataframe, scipy sparse matrix), the
        regular return value is returned.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        y : array-like
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
        y : array-like
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


class Incremental(ParallelPostFit):
    """Metaestimator for feeding Dask Arrays to an estimator blockwise.

    This wrapper provides a bridge between Dask objects and estimators
    implementing the ``partial_fit`` API. These *incremental learners* can
    train on batches of data. This fits well with Dask's blocked data
    structures.

    See the `list of incremental learners`_ in the scikit-learn documentation
    for a list of estimators that implement the ``partial_fit`` API. Note that
    `Incremental` is not limited to just these classes, it will work on any
    estimator implementing ``partial_fit``, including those defined outside of
    scikit-learn itself.

    Calling :meth:`Incremental.fit` with a Dask Array will pass each block of
    the Dask array or arrays to ``estimator.partial_fit`` *sequentially*.

    Like :class:`ParallelPostFit`, the methods available after fitting (e.g.
    :meth:`Incremental.predict`, etc.) are all parallel and delayed.

    .. _list of incremental learners: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning  # noqa

    Parameters
    ----------
    estimator : Estimator
        Any object supporting the scikit-learn `parital_fit` API.
    **kwargs
        Additional keyword arguments passed through the the underlying
        estimator's `partial_fit` method.

    Examples
    --------
    >>> from dask_ml.wrappers import Incremental
    >>> from dask_ml.datasets import make_classification
    >>> import sklearn.linear_model
    >>> X, y = make_classification(chunks=25)
    >>> est = sklearn.linear_model.SGDClassifier()
    >>> clf = Incremental(est, classes=[0, 1])
    >>> clf.fit(X, y)
    """
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        self.fit_kwargs = kwargs

    def fit(self, X, y=None):
        from ._partial import fit

        fit_kwargs = self.fit_kwargs or {}
        result = fit(self.estimator, X, y, **fit_kwargs)

        # Copy the learned attributes over to self
        attrs = {k: v for k, v in vars(result).items() if k.endswith('_')}
        for k, v in attrs.items():
            setattr(self, k, v)
        return self


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
