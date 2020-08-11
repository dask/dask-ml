import sklearn.base


class ClassifierMixin(sklearn.base.ClassifierMixin):
    """Mixin class for all classifiers in scikit-learn."""

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        This matches the scikit-learn implementation with the difference
        that :meth:`dask_ml.metrics.accuracy_score` is used rather than
        :meth:`sklearn.metrics.accuracy_score`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class RegressorMixin(sklearn.base.RegressorMixin):
    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        This matches the scikit-learn implementation with the differences
        that

        * :meth:`dask_ml.metrics.accuracy_score` is used rather than
          :meth:`sklearn.metrics.accuracy_score`.
        * The ``'uniform_average'`` method is used for multioutput results
          rather than ``'variance_weighted'``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(
            y, y_pred, sample_weight=sample_weight, multioutput="uniform_average"
        )
