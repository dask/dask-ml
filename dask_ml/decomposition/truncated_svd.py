import dask.array as da
from dask import compute
from sklearn.base import BaseEstimator, TransformerMixin

from .._compat import DASK_2_26_0
from ..utils import svd_flip


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        algorithm="tsqr",
        n_iter=5,
        random_state=None,
        tol=0.0,
        compute=True,
    ):
        """Dimensionality reduction using truncated SVD (aka LSA).

        This transformer performs linear dimensionality reduction by means of
        truncated singular value decomposition (SVD). Contrary to PCA, this
        estimator does not center the data before computing the singular value
        decomposition.

        Parameters
        ----------
        n_components : int, default = 2
            Desired dimensionality of output data.
            Must be less than or equal to the number of features.
            The default value is useful for visualization.

        algorithm : {'tsqr', 'randomized'}
            SVD solver to use. Both use the `tsqr` (for "tall-and-skinny QR")
            algorithm internally. 'randomized' uses an approximate algorithm
            that is faster, but not exact. See the References for more.

        n_iter : int, optional (default 0)
            Number of power iterations, useful when the singular values
            decay slowly. Error decreases exponentially as n_power_iter
            increases. In practice, set n_power_iter <= 4.

        random_state : int, RandomState instance or None, optional
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState instance
            used by `np.random`.

        tol : float, optional
            Ignored.

        compute : bool
            Whether or not SVD results should be computed
            eagerly, by default True.

        Attributes
        ----------
        components_ : array, shape (n_components, n_features)

        explained_variance_ : array, shape (n_components,)
            The variance of the training samples transformed by a projection to
            each component.

        explained_variance_ratio_ : array, shape (n_components,)
            Percentage of variance explained by each of the selected
            components.

        singular_values_ : array, shape (n_components,)
            The singular values corresponding to each of the selected
            components. The singular values are equal to the 2-norms of the
            ``n_components`` variables in the lower-dimensional space.

        See Also
        --------
        dask.array.linalg.tsqr
        dask.array.linalg.svd_compressed

        References
        ----------

        Direct QR factorizations for tall-and-skinny matrices in
        MapReduce architectures.
        A. Benson, D. Gleich, and J. Demmel.
        IEEE International Conference on Big Data, 2013.
        http://arxiv.org/abs/1301.1071

        Notes
        -----
        SVD suffers from a problem called "sign indeterminacy", which means
        the sign of the ``components_`` and the output from transform depend on
        the algorithm and random state. To work around this, fit instances of
        this class to data once, then keep the instance around to do
        transformations.

        .. warning::

           The implementation currently does not support sparse matricies.

        Examples
        --------
        >>> from dask_ml.decomposition import TruncatedSVD
        >>> import dask.array as da
        >>> X = da.random.normal(size=(1000, 20), chunks=(100, 20))
        >>> svd = TruncatedSVD(n_components=5, n_iter=3, random_state=42)
        >>> svd.fit(X)  # doctest: +NORMALIZE_WHITESPACE
        TruncatedSVD(algorithm='tsqr', n_components=5, n_iter=3,
                     random_state=42, tol=0.0)

        >>> print(svd.explained_variance_ratio_)  # doctest: +ELLIPSIS
        [0.06386323 0.06176776 0.05901293 0.0576399  0.05726607]
        >>> print(svd.explained_variance_ratio_.sum())  # doctest: +ELLIPSIS
        0.299...
        >>> print(svd.singular_values_)  # doctest: +ELLIPSIS
        array([35.92469517, 35.32922121, 34.53368856, 34.138..., 34.013...])

        Note that ``tranform`` returns a ``dask.Array``.

        >>> svd.transform(X)
        dask.array<sum-agg, shape=(1000, 5), dtype=float64, chunksize=(100, 5)>
        """
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.compute = compute

    def fit(self, X, y=None):
        """Fit truncated SVD on training data X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        return self

    def _check_array(self, X):
        if self.n_components >= X.shape[1]:
            raise ValueError(
                "n_components must be < n_features; "
                "got {} >= {}".format(self.n_components, X.shape[1])
            )
        return X

    def fit_transform(self, X, y=None):
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array, of the
            same type as the input array. If ``X`` was a ``dask.array``, then
            ``X_new`` will be a ``dask.array`` with the same chunks along the
            first dimension.
        """
        X = self._check_array(X)
        if self.algorithm not in {"tsqr", "randomized"}:
            raise ValueError(
                "`algorithm` must be 'tsqr' or 'randomzied', not '{}'".format(
                    self.algorithm
                )
            )
        if self.algorithm == "tsqr":
            u, s, v = da.linalg.svd(X)
            u = u[:, : self.n_components]
            s = s[: self.n_components]
            v = v[: self.n_components]
        else:
            u, s, v = da.linalg.svd_compressed(
                X, self.n_components, self.n_iter, seed=self.random_state
            )
        if not DASK_2_26_0:
            u, v = svd_flip(u, v)

        X_transformed = u * s
        explained_var = X_transformed.var(axis=0)
        full_var = X.var(axis=0).sum()
        explained_variance_ratio = explained_var / full_var

        if self.compute:
            v, explained_var, explained_variance_ratio, s = compute(
                v, explained_var, explained_variance_ratio, s
            )
        self.components_ = v
        self.explained_variance_ = explained_var
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = s
        self.n_features_in_ = X.shape[1]
        return X_transformed

    def transform(self, X, y=None):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
             Data to be transformed.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array, of the
            same type as the input array. If ``X`` was a ``dask.array``, then
            ``X_new`` will be a ``dask.array`` with the same chunks along the
            first dimension.
        """
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        # X = check_array(X)
        return X.dot(self.components_)
