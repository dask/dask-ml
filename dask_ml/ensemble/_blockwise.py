import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.base
from sklearn.utils.validation import check_is_fitted

from ..base import ClassifierMixin, RegressorMixin


class BlockwiseBase(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def _check_array(self, X):
        return X

    def fit(self, X, y, **kwargs):
        X = self._check_array(X)
        estimatord = dask.delayed(self.estimator)

        Xs = X.to_delayed()
        ys = y.to_delayed()
        if isinstance(X, da.Array):
            Xs = Xs.flatten()
        if isinstance(y, da.Array):
            ys = ys.flatten()

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
            dtype = "int"
        else:
            dtype = "float"

        if isinstance(X, da.Array):
            results = [
                X.map_blocks(_predict, dtype=dtype, estimator=estimator, drop_axis=1)
                for estimator in self.estimators_
            ]
            combined = da.vstack(results).T.rechunk({1: -1})
        elif isinstance(X, dd._Frame):
            results = [
                X.map_partitions(
                    _predict, estimator=estimator, meta=np.array([], dtype=dtype)
                )
                for estimator in self.estimators_
            ]
            combined = da.vstack(results, allow_unknown_chunksizes=True).T.rechunk(
                {1: -1}
            )
        else:
            combined = np.vstack(
                [estimator.predict(X) for estimator in self.estimators_]
            ).T

        return combined


class BlockwiseVotingClassifier(ClassifierMixin, BlockwiseBase):
    """
    Blockwise training and ensemble voting classifier.

    This classifier trains on blocks / partitions of Dask Arrays or DataFrames.
    A cloned version of `estimator` will be fit *independently* on each block
    or partition of the Dask collection.

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
        check_is_fitted(self)
        # TODO: check for just row-wise partition!
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'hard' voting
            predictions = self._predict(X)  # (N, n_estimators)  ensure chunking!
            if isinstance(predictions, da.Array):
                maj = predictions.map_blocks(_vote_block, dtype="int", drop_axis=1)
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
        check_is_fitted(self)
        avg = np.average(self._collect_probas(X), axis=0)
        return avg

    def _collect_probas(self, X):
        if isinstance(X, da.Array):
            chunks = X.chunks[0], len(self.classes_)
            probas = [
                X.map_blocks(
                    _predict_proba, dtype="float", estimator=estimator, chunks=chunks
                )
                for estimator in self.estimators_
            ]
            combined = da.stack(probas)  # (n_estimators, len(X), n_classses)
        elif isinstance(X, dd._Frame):
            meta = np.empty((0, len(self.classes_)), dtype="float")
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
    """

    def predict(self, X):
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1)


def fit(estimator, x, y):
    # TODO: logging
    estimator.fit(x, y)
    return estimator


def _predict(part, estimator):
    return estimator.predict(part)


def _predict_proba(part, estimator):
    return estimator.predict_proba(part)


def _vote(x):
    return np.argmax(np.bincount(x))


def _vote_block(block):
    return np.apply_along_axis(_vote, 1, block)
