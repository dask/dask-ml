import dask.array as da
import numpy as np
from dask import delayed
from sklearn import naive_bayes as _naive_bayes
from sklearn.base import BaseEstimator

from ._partial import _BigPartialFitMixin, _copy_partial_doc


class GaussianNB(BaseEstimator):
    """
    Fit a naive bayes model with a Gaussian likelihood

    Examples
    --------
    >>> from dask_ml import datasets
    >>> from dask_ml.naive_bayes import GaussianNB
    >>> X, y = datasets.make_classification(chunks=10)
    >>> gnb = GaussianNB()
    >>> gnb.fit(X, y)
    """

    def __init__(self, priors=None, classes=None):
        self.priors = priors
        self.classes = classes
        self.classes_ = classes
        self.class_prior_ = None
        self.class_count_ = None
        self.theta_ = None
        self.sigma_ = None

    def fit(self, X, y=None):
        if self.classes is None:
            # TODO: delayed
            self.classes_ = np.unique(y)

        thetas = []
        sigmas = []
        counts = []
        N, P = X.shape
        K = len(self.classes_)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            thetas.append(X_c.mean(axis=0))
            sigmas.append(X_c.var(axis=0))
            counts.append(delayed(len)(X_c))

        thetas = da.from_delayed(delayed(np.array)(thetas), (K, P), np.float64)
        sigmas = da.from_delayed(delayed(np.array)(sigmas), (K, P), np.float64)
        counts = da.from_delayed(
            delayed(np.array)(counts, np.float64), (P,), np.float64
        )
        priors = counts / N

        # Should these be explicitly cached on self?
        self.theta_ = thetas
        self.sigma_ = sigmas
        self.class_count_ = counts
        self.class_prior_ = priors

        return self

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return delayed(self.classes_)[da.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - log_prob_x.reshape(-1, 1)

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        return da.exp(self.predict_log_proba(X))

    def _joint_log_likelihood(self, X):
        jll = []
        for i in range(np.size(self.classes_)):
            jointi = da.log(self.class_prior_[i])
            n_ij = -0.5 * da.sum(da.log(2.0 * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * da.sum(
                ((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1
            )
            jll.append(jointi + n_ij)

        joint_log_likelihood = da.stack(jll).T
        return joint_log_likelihood


@_copy_partial_doc
class PartialMultinomialNB(_BigPartialFitMixin, _naive_bayes.MultinomialNB):
    _init_kwargs = ["classes"]
    _fit_kwargs = ["classes"]


@_copy_partial_doc
class PartialBernoulliNB(_BigPartialFitMixin, _naive_bayes.BernoulliNB):
    _init_kwargs = ["classes"]
    _fit_kwargs = ["classes"]


def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    if axis == 0:
        pass
    elif axis == 1:
        arr = arr.T
    else:
        raise NotImplementedError
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = da.log(da.sum(da.exp(arr - vmax), axis=0))
    out += vmax
    return out
