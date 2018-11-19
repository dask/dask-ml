"""Meta-estimators for parallelizing estimators using the scikit-learn API."""
import logging

import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import sklearn.base
import sklearn.metrics
from sklearn.utils.validation import check_is_fitted

from dask_ml.utils import _timer

from ._partial import fit
from ._utils import copy_learned_attributes
from .metrics import check_scoring, get_scorer

logger = logging.getLogger(__name__)


class ParallelPostFit(sklearn.base.BaseEstimator, sklearn.base.MetaEstimatorMixin):
    """Meta-estimator for parallel predict and transform.

    Parameters
    ----------
    estimator : Estimator
        The underlying estimator that is fit.

    scoring : string or callable, optional
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique)
        strings or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        .. warning::

           If None, the estimator's default scorer (if available) is used.
           Most scikit-learn estimators will convert large Dask arrays to
           a single NumPy array, which may exhaust the memory of your worker.
           You probably want to always specify `scoring`.

    Notes
    -----

    .. warning::

       This class is not appropriate for parallel or distributed *training*
       on large datasets. For that, see :class:`Incremental`, which provides
       distributed (but sequential) training. If you're doing distributed
       hyperparameter optimization on larger-than-memory datasets, see
       :class:`dask_ml.model_selection.IncrementalSearch`.

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

    See Also
    --------
    Incremental
    dask_ml.model_selection.IncrementalSearch

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> import sklearn.datasets
    >>> import dask_ml.datasets

    Make a small 1,000 sample 2 training dataset and fit normally.

    >>> X, y = sklearn.datasets.make_classification(n_samples=1000,
    ...                                             random_state=0)
    >>> clf = ParallelPostFit(estimator=GradientBoostingClassifier(),
    ...                       scoring='accuracy')
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

    def __init__(self, estimator=None, scoring=None):
        self.estimator = estimator
        self.scoring = scoring

    def _check_array(self, X):
        """Validate an array for post-fit tasks.

        Parameters
        ----------
        X : Union[Array, DataFrame]

        Returns
        -------
        same type as 'X'

        Notes
        -----
        The following checks are applied.

        - Ensure that the array is blocked only along the samples.
        """
        if isinstance(X, da.Array):
            if X.ndim == 2 and X.numblocks[1] > 1:
                logger.debug("auto-rechunking 'X'")
                if not np.isnan(X.chunks[0]).any():
                    X = X.rechunk({0: "auto", 1: -1})
                else:
                    X = X.rechunk({1: -1})
        return X

    @property
    def _postfit_estimator(self):
        # The estimator instance to use for postfit tasks like score
        return self.estimator

    def fit(self, X, y=None, **kwargs):
        """Fit the underlying estimator.

        Parameters
        ----------
        X, y : array-like
        **kwargs
            Additional fit-kwargs for the underlying estimator.

        Returns
        -------
        self : object
        """
        logger.info("Starting fit")
        with _timer("fit", _logger=logger):
            result = self.estimator.fit(X, y, **kwargs)

        # Copy over learned attributes
        copy_learned_attributes(result, self)
        copy_learned_attributes(result, self.estimator)
        return self

    def partial_fit(self, X, y=None, **kwargs):
        logger.info("Starting partial_fit")
        with _timer("fit", _logger=logger):
            result = self.estimator.partial_fit(X, y, **kwargs)

        # Copy over learned attributes
        copy_learned_attributes(result, self)
        copy_learned_attributes(result, self.estimator)
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
        self._check_method("transform")
        X = self._check_array(X)

        if isinstance(X, da.Array):
            return X.map_blocks(_transform, estimator=self._postfit_estimator)
        elif isinstance(X, dd._Frame):
            return X.map_partitions(_transform, estimator=self._postfit_estimator)
        else:
            return _transform(X, estimator=self._postfit_estimator)

    def score(self, X, y, compute=True):
        """Returns the score on the given data.

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
        scoring = self.scoring
        X = self._check_array(X)
        y = self._check_array(y)

        if not scoring:
            if type(self._postfit_estimator).score == sklearn.base.RegressorMixin.score:
                scoring = "r2"
            elif (
                type(self._postfit_estimator).score
                == sklearn.base.ClassifierMixin.score
            ):
                scoring = "accuracy"
        else:
            scoring = self.scoring

        if scoring:
            if not dask.is_dask_collection(X) and not dask.is_dask_collection(y):
                scorer = sklearn.metrics.get_scorer(scoring)
            else:
                scorer = get_scorer(scoring, compute=compute)
            return scorer(self, X, y)
        else:
            return self._postfit_estimator.score(X, y)

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
        self._check_method("predict")
        X = self._check_array(X)

        if isinstance(X, da.Array):
            result = X.map_blocks(
                _predict, dtype="int", estimator=self._postfit_estimator, drop_axis=1
            )
            return result

        elif isinstance(X, dd._Frame):
            return X.map_partitions(
                _predict, estimator=self._postfit_estimator, meta=np.array([1])
            )

        else:
            return _predict(X, estimator=self._postfit_estimator)

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
        X = self._check_array(X)

        self._check_method("predict_proba")

        if isinstance(X, da.Array):
            # XXX: multiclass
            return X.map_blocks(
                _predict_proba,
                estimator=self._postfit_estimator,
                dtype="float",
                chunks=(X.chunks[0], len(self.classes_)),
            )
        elif isinstance(X, dd._Frame):
            return X.map_partitions(_predict_proba, estimator=self._postfit_estimator)
        else:
            return _predict_proba(X, estimator=self._postfit_estimator)

    def _check_method(self, method):
        """Check if self.estimator has 'method'.

        Raises
        ------
        AttributeError
        """
        estimator = self._postfit_estimator
        if not hasattr(estimator, method):
            msg = "The wrapped estimator '{}' does not have a '{}' method.".format(
                estimator, method
            )
            raise AttributeError(msg)
        return getattr(estimator, method)


class Incremental(ParallelPostFit):
    """Metaestimator for feeding Dask Arrays to an estimator blockwise.

    This wrapper provides a bridge between Dask objects and estimators
    implementing the ``partial_fit`` API. These *incremental learners* can
    train on batches of data. This fits well with Dask's blocked data
    structures.

    .. note::

       This meta-estimator is not appropriate for hyperparameter optimization
       on larger-than-memory datasets. For that, see
       :class:dask_ml.model_selection.IncrementalSearch`.

    See the `list of incremental learners`_ in the scikit-learn documentation
    for a list of estimators that implement the ``partial_fit`` API. Note that
    `Incremental` is not limited to just these classes, it will work on any
    estimator implementing ``partial_fit``, including those defined outside of
    scikit-learn itself.

    Calling :meth:`Incremental.fit` with a Dask Array will pass each block of
    the Dask array or arrays to ``estimator.partial_fit`` *sequentially*.

    Like :class:`ParallelPostFit`, the methods available after fitting (e.g.
    :meth:`Incremental.predict`, etc.) are all parallel and delayed.

    The ``estimator_`` attribute is a clone of `estimator` that was actually
    used during the call to ``fit``. All attributes learned during training
    are available on ``Incremental`` directly.

    .. _list of incremental learners: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning  # noqa

    Parameters
    ----------
    estimator : Estimator
        Any object supporting the scikit-learn ``parital_fit`` API.

    scoring : string or callable, optional
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique)
        strings or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        .. warning::

           If None, the estimator's default scorer (if available) is used.
           Most scikit-learn estimators will convert large Dask arrays to
           a single NumPy array, which may exhaust the memory of your worker.
           You probably want to always specify `scoring`.

    random_state : int or numpy.random.RandomState, optional
        Random object that determines how to shuffle blocks.

    shuffle_blocks : bool, default True
        Determines whether to call ``partial_fit`` on a randomly selected chunk
        of the Dask arrays (default), or to fit in sequential order. This does
        not control shuffle between blocks or shuffling each block.

    Attributes
    ----------
    estimator_ : Estimator
        A clone of `estimator` that was actually fit during the ``.fit`` call.

    See Also
    --------
    ParallelPostFit
    dask_ml.model_selection.IncrementalSearch

    Examples
    --------
    >>> from dask_ml.wrappers import Incremental
    >>> from dask_ml.datasets import make_classification
    >>> import sklearn.linear_model
    >>> X, y = make_classification(chunks=25)
    >>> est = sklearn.linear_model.SGDClassifier()
    >>> clf = Incremental(est, scoring='accuracy')
    >>> clf.fit(X, y, classes=[0, 1])

    When used inside a grid search, prefix the underlying estimator's
    parameter names with ``estimator__``.

    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = {"estimator__alpha": [0.1, 1.0, 10.0]}
    >>> gs = GridSearchCV(clf, param_grid)
    >>> gs.fit(X, y, classes=[0, 1])
    """

    def __init__(
        self, estimator=None, scoring=None, shuffle_blocks=True, random_state=None
    ):
        self.shuffle_blocks = shuffle_blocks
        self.random_state = random_state
        super(Incremental, self).__init__(estimator=estimator, scoring=scoring)

    @property
    def _postfit_estimator(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_

    def _fit_for_estimator(self, estimator, X, y, **fit_kwargs):
        check_scoring(estimator, self.scoring)
        if not dask.is_dask_collection(X) and not dask.is_dask_collection(y):
            result = estimator.partial_fit(X=X, y=y, **fit_kwargs)
        else:
            result = fit(
                estimator,
                X,
                y,
                random_state=self.random_state,
                shuffle_blocks=self.shuffle_blocks,
                **fit_kwargs
            )

        copy_learned_attributes(result, self)
        self.estimator_ = result
        return self

    def fit(self, X, y=None, **fit_kwargs):
        estimator = sklearn.base.clone(self.estimator)
        self._fit_for_estimator(estimator, X, y, **fit_kwargs)
        return self

    def partial_fit(self, X, y=None, **fit_kwargs):
        """Fit the underlying estimator.

        If this estimator has not been previously fit, this is identical to
        :meth:`Incremental.fit`. If it has been previously fit,
        ``self.estimator_`` is used as the starting point.

        Parameters
        ----------
        X, y : array-like
        **kwargs
            Additional fit-kwargs for the underlying estimator.

        Returns
        -------
        self : object
        """
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            estimator = sklearn.base.clone(self.estimator)
        return self._fit_for_estimator(estimator, X, y, **fit_kwargs)


def _first_block(dask_object):
    """Extract the first block / partition from a dask object
    """
    if isinstance(dask_object, da.Array):
        if dask_object.ndim > 1 and dask_object.numblocks[-1] != 1:
            raise NotImplementedError(
                "IID estimators require that the array "
                "blocked only along the first axis. "
                "Rechunk your array before fitting."
            )
        shape = (dask_object.chunks[0][0],)
        if dask_object.ndim > 1:
            shape = shape + (dask_object.chunks[1][0],)

        return da.from_delayed(
            dask_object.to_delayed().flatten()[0], shape, dask_object.dtype
        )

    if isinstance(dask_object, dd._Frame):
        return dask_object.get_partition(0)

    else:
        return dask_object


def _predict(part, estimator):
    return estimator.predict(part)


def _predict_proba(part, estimator):
    return estimator.predict_proba(part)


def _transform(part, estimator):
    return estimator.transform(part)
