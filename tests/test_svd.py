"""Test truncated SVD transformer."""
import dask.array as da
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from sklearn import decomposition as sd
from sklearn.utils import check_random_state

from dask_ml import decomposition as dd
from dask_ml.utils import assert_estimator_equal, flip_vector_signs

# Make an X that looks somewhat like a small tf-idf matrix.
# XXX newer versions of SciPy have scipy.sparse.rand for this.
shape = 60, 55
n_samples, n_features = shape
rng = check_random_state(42)
X = rng.randint(-100, 20, np.product(shape)).reshape(shape)
X = sp.csr_matrix(np.maximum(X, 0), dtype=np.float64)
X.data[:] = 1 + np.log(X.data)
Xdense = X.A

dXdense = da.from_array(Xdense, chunks=(30, 55))


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_basic(algorithm):
    a = dd.TruncatedSVD(random_state=0, algorithm=algorithm)
    b = sd.TruncatedSVD(random_state=0)
    b.fit(Xdense)
    a.fit(dXdense)

    np.testing.assert_allclose(
        flip_vector_signs(a.components_, axis=1),
        flip_vector_signs(b.components_, axis=1),
        atol=1e-3,
    )
    assert_estimator_equal(
        a, b, exclude=["components_", "explained_variance_"], atol=1e-3
    )
    assert a.explained_variance_.shape == b.explained_variance_.shape
    np.testing.assert_allclose(a.explained_variance_, b.explained_variance_, rtol=0.01)


# The rest come straight from scikit-learn, with dask arrays substituted


def test_algorithms():
    svd_a = sd.TruncatedSVD(30, algorithm="arpack")
    svd_r = dd.TruncatedSVD(30, algorithm="tsqr", random_state=42)

    Xa = svd_a.fit_transform(Xdense)[:, :6]
    Xr = svd_r.fit_transform(dXdense)[:, :6]
    assert_array_almost_equal(
        flip_vector_signs(Xa, axis=0), flip_vector_signs(Xr, axis=0), decimal=5
    )

    comp_a = np.abs(svd_a.components_)
    comp_r = np.abs(svd_r.components_)
    # All elements are equal, but some elements are more equal than others.
    assert_array_almost_equal(comp_a[:9], comp_r[:9])
    assert_array_almost_equal(comp_a[9:], comp_r[9:], decimal=2)


def test_attributes():
    for n_components in (10, 25, 41):
        tsvd = dd.TruncatedSVD(n_components).fit(dXdense)
        assert tsvd.n_components == n_components
        assert tsvd.components_.shape == (n_components, n_features)


@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
@pytest.mark.parametrize("compute", [True, False])
def test_compute(algorithm, compute):
    est = dd.TruncatedSVD(random_state=0, algorithm=algorithm, compute=compute)
    est.fit(dXdense)
    array_class = np.ndarray if compute else da.Array
    assert isinstance(est.components_, array_class)
    assert isinstance(est.explained_variance_, array_class)
    assert isinstance(est.explained_variance_ratio_, array_class)
    assert isinstance(est.singular_values_, array_class)


def test_too_many_components():
    for n_components in (n_features, n_features + 1):
        tsvd = dd.TruncatedSVD(n_components=n_components)
        with pytest.raises(ValueError):
            tsvd.fit(dXdense)


def test_inverse_transform():
    # We need a lot of components for the reconstruction to be "almost
    # equal" in all positions. XXX Test means or sums instead?
    a = dd.TruncatedSVD(n_components=52, random_state=42, n_iter=5)
    b = sd.TruncatedSVD(n_components=52, random_state=42)
    b.fit(Xdense)
    Xt = a.fit_transform(dXdense)
    Xinv = a.inverse_transform(Xt)
    assert_array_almost_equal(Xinv.compute(), Xdense, decimal=1)


def test_integers():
    Xint = dXdense.astype(np.int64)
    tsvd = dd.TruncatedSVD(n_components=6)
    Xtrans = tsvd.fit_transform(Xint)
    Xtrans.shape == (n_samples, tsvd.n_components)


def test_singular_values():
    # Check that the TruncatedSVD output has the correct singular values

    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80

    X = rng.randn(n_samples, n_features)
    dX = da.from_array(X, chunks=(n_samples // 2, n_features))

    apca = dd.TruncatedSVD(n_components=2, algorithm="tsqr", random_state=rng).fit(dX)
    rpca = sd.TruncatedSVD(n_components=2, algorithm="arpack", random_state=rng).fit(X)
    assert_array_almost_equal(apca.singular_values_, rpca.singular_values_, 12)

    # Compare to the Frobenius norm
    X_apca = apca.transform(X)
    X_rpca = rpca.transform(X)
    assert_array_almost_equal(
        np.sum(apca.singular_values_ ** 2.0), np.linalg.norm(X_apca, "fro") ** 2.0, 12
    )
    assert_array_almost_equal(
        np.sum(rpca.singular_values_ ** 2.0), np.linalg.norm(X_rpca, "fro") ** 2.0, 12
    )

    # Compare to the 2-norms of the score vectors
    assert_array_almost_equal(
        apca.singular_values_, np.sqrt(np.sum(X_apca ** 2.0, axis=0)), 12
    )
    assert_array_almost_equal(
        rpca.singular_values_, np.sqrt(np.sum(X_rpca ** 2.0, axis=0)), 12
    )

    # Set the singular values and see what we get back
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110

    X = rng.randn(n_samples, n_features)
    dX = da.from_array(X, chunks=(50, n_features))

    apca = dd.TruncatedSVD(n_components=3, algorithm="randomized", random_state=0)
    rpca = sd.TruncatedSVD(n_components=3, algorithm="randomized", random_state=0)
    X_apca = apca.fit_transform(dX).compute()
    X_rpca = rpca.fit_transform(X)

    X_apca /= np.sqrt(np.sum(X_apca ** 2.0, axis=0))
    X_rpca /= np.sqrt(np.sum(X_rpca ** 2.0, axis=0))
    X_apca[:, 0] *= 3.142
    X_apca[:, 1] *= 2.718
    X_rpca[:, 0] *= 3.142
    X_rpca[:, 1] *= 2.718

    X_hat_apca = np.dot(X_apca, apca.components_)
    X_hat_rpca = np.dot(X_rpca, rpca.components_)
    apca.fit(da.from_array(X_hat_apca, chunks=(50, n_features)))
    rpca.fit(X_hat_rpca)
    assert_array_almost_equal(apca.singular_values_, [3.142, 2.718, 1.0], 14)
    assert_array_almost_equal(rpca.singular_values_, [3.142, 2.718, 1.0], 14)
