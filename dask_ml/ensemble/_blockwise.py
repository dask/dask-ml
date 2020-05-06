import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.base
from sklearn.utils.validation import check_is_fitted

from ..base import ClassifierMixin, RegressorMixin
from ..utils import check_array


class BlockwiseBase(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def _check_array(self, X):
        return check_array(
            X,
            accept_dask_dataframe=True,
            accept_unknown_chunks=True,
            preserve_pandas_dataframe=True,
        )

    def fit(self, X, y, **kwargs):
        X = self._check_array(X)
        estimatord = dask.delayed(self.estimator)

        Xs = X.to_delayed()
        ys = y.to_delayed()
        if isinstance(X, da.Array):
            Xs = Xs.flatten()
        if isinstance(y, da.Array):
            ys = ys.flatten()

        if len(Xs) != len(ys):
            raise ValueError(
                f"The number of blocks in X and y must match. {len(Xs)} != {len(ys)}"
            )

        estimators = [
            dask.delayed(sklearn.base.clone)(estimatord) for _ in range(len(Xs))
        ]
        results = [
            estimator_.fit(X_, y_, **kwargs)
            for estimator_, X_, y_, in zip(estimators, Xs, ys)
        ]
        results = list(dask.compute(*results))
        self.estimators_ = results

    def _predict(self, X):
        """Collect results from many predict calls"""
        if isinstance(self, ClassifierMixin):
            dtype = "int64"
        else:
            dtype = "float64"

        if isinstance(X, da.Array):
            chunks = (X.chunks[0], len(self.estimators_))
            combined = X.map_blocks(
                _predict_stack,
                estimators=self.estimators_,
                dtype=np.dtype(dtype),
                chunks=chunks,
            )
        elif isinstance(X, dd._Frame):
            meta = np.empty((0, len(self.classes_)), dtype=dtype)
            combined = X.map_partitions(
                _predict_stack, estimators=self.estimators_, meta=meta
            )
        else:
            # TODO: this should be done in parallel?
            combined = np.vstack(
                [estimator.predict(X) for estimator in self.estimators_]
            ).T

        return combined


class BlockwiseVotingClassifier(ClassifierMixin, BlockwiseBase):
    """
    Blockwise training and ensemble voting classifier.

    This classifier trains on blocks / partitions of Dask Arrays or DataFrames.
    A cloned version of `estimator` will be fit *independently* on each block
    or partition of the Dask collection. This is useful when the sub estimator
    only works on small in-memory data structures like a NumPy array or pandas
    DataFrame.

    Prediction is done by the *ensemble* of learned models.

    .. warning::

       Ensure that your data are sufficiently shuffled prior to training!
       If the values of the various blocks / partitions of your dataset are not
       distributed similarly, the classifier will give poor results.

    Parameters
    ----------
    estimator : Estimator
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    classes : list-like, optional
        The set of classes that `y` can take. This can also be provided as
        a fit param if the underlying estimator requires `classes` at fit time.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators that are `estimator` fitted
        on each partition / block of the inputs.

    classes_ : array-like, shape (n_predictions,)
        The class labels.

    Examples
    --------
    >>> import dask_ml.datasets
    >>> import dask_ml.ensemble
    >>> import sklearn.linear_model
    >>> X, y = dask_ml.datasets.make_classification(n_samples=100_000,
    >>> ...                                         chunks=10_000)
    >>> subestimator = sklearn.linear_model.RidgeClassifier(random_state=0)
    >>> clf = dask_ml.ensemble.BlockwiseVotingClassifier(
    >>> ...     subestimator,
    >>> ...     classes=[0, 1]
    >>> ... )
    >>> clf.fit(X, y)
    """

    def __init__(self, estimator, voting="hard", classes=None):
        self.voting = voting
        self.classes = classes
        super().__init__(estimator)

    def fit(self, X, y, **kwargs):
        if self.classes is None and "classes" not in kwargs:
            raise ValueError("Must provide the classes of `y`.")
        elif self.classes is not None:
            classes = self.classes
        else:
            classes = kwargs["classes"]
        super().fit(X, y, **kwargs)
        self.classes_ = np.array(classes)

    def predict(self, X):
        check_is_fitted(self, attributes=["estimators_"])
        X = self._check_array(X)
        # TODO: check for just row-wise partition!
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'hard' voting
            predictions = self._predict(X)  # (N, n_estimators)  ensure chunking!
            if isinstance(predictions, da.Array):
                maj = predictions.map_blocks(_vote_block, dtype="int64", drop_axis=1)
            else:
                maj = _vote_block(predictions)
        return maj

    @property
    def predict_proba(self):
        if self.voting == "hard":
            raise AttributeError(
                "predict_proba is not available when" " voting=%r" % self.voting
            )
        return self._predict_proba

    def _predict_proba(self, X):
        check_is_fitted(self, attributes=["estimators_"])
        X = self._check_array(X)
        avg = np.average(self._collect_probas(X), axis=0)
        return avg

    def _collect_probas(self, X):
        if isinstance(X, da.Array):
            chunks = (len(self.estimators_), X.chunks[0], len(self.classes_))
            meta = np.array([], dtype="float64")
            # (n_estimators, len(X), n_classses)
            combined = X.map_blocks(
                _predict_proba_stack,
                estimators=self.estimators_,
                chunks=chunks,
                meta=meta,
            )
        elif isinstance(X, dd._Frame):
            # TODO: replace with a _predict_proba_stack version.
            # This current raises; dask.dataframe doesn't like map_partitions that
            # return new axes.
            # meta = np.empty((len(self.estimators_), 0, len(self.classes_)),
            #                 dtype="float64")
            # combined = X.map_partitions(_predict_proba_stack, meta=meta,
            #                             estimators=self.estimators_)
            # combined._chunks = ((len(self.estimators_),),
            #                     (np.nan,) * X.npartitions,
            #                     (len(X.columns),))
            meta = np.empty((0, len(self.classes_)), dtype="float64")
            probas = [
                X.map_partitions(_predict_proba, meta=meta, estimator=estimator)
                for estimator in self.estimators_
            ]
            # TODO(https://github.com/dask/dask/issues/6177): replace with da.stack
            chunks = probas[0]._chunks
            for proba in probas:
                proba._chunks = ((1,) * len(chunks[0]), chunks[1])

            combined = da.stack(probas)
            combined._chunks = ((1,) * len(self.estimators_),) + chunks
        else:
            # ndarray, etc.
            combined = np.stack(
                [estimator.predict_proba(X) for estimator in self.estimators_]
            )

        return combined


class BlockwiseVotingRegressor(RegressorMixin, BlockwiseBase):
    """
    Blockwise training and ensemble voting regressor.

    This regressor trains on blocks / partitions of Dask Arrays or DataFrames.
    A cloned version of `estimator` will be fit *independently* on each block
    or partition of the Dask collection.

    Prediction is done by the *ensemble* of learned models.

    .. warning::

       Ensure that your data are sufficiently shuffled prior to training!
       If the values of the various blocks / partitions of your dataset are not
       distributed similarly, the regressor will give poor results.

    Parameters
    ----------
    estimator : Estimator

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators that are `estimator` fitted
        on each partition / block of the inputs.

    Examples
    --------
    >>> import dask_ml.datasets
    >>> import dask_ml.ensemble
    >>> import sklearn.linear_model
    >>> X, y = dask_ml.datasets.make_regression(n_samples=100_000,
    ...                                         chunks=10_000)
    >>> subestimator = sklearn.linear_model.LinearRegression()
    >>> clf = dask_ml.ensemble.BlockwiseVotingRegressor(
    ...     subestimator,
    ... )
    >>> clf.fit(X, y)
    """

    def predict(self, X):
        check_is_fitted(self, attributes=["estimators_"])
        return np.average(self._predict(X), axis=1)


def fit(estimator, x, y):
    # TODO: logging
    estimator.fit(x, y)
    return estimator


def _predict_proba(part, estimator):
    return estimator.predict_proba(part)


def _vote(x):
    return np.argmax(np.bincount(x))


def _vote_block(block):
    return np.apply_along_axis(_vote, 1, block)


def _predict_stack(part, estimators):
    # predict for a batch of estimators and stack up the results.
    batches = [estimator.predict(part) for estimator in estimators]
    return np.vstack(batches).T


def _predict_proba_stack(part, estimators):
    # predict for a batch of estimators and stack up the results.
    batches = [estimator.predict_proba(part) for estimator in estimators]
    return np.stack(batches)
