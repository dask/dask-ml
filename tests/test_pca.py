from itertools import product

import dask.array as da
import dask.dataframe
import numpy as np
import pandas as pd
import pytest
import scipy as sp
import sklearn.decomposition as sd
from dask.array.utils import assert_eq
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_raises
from sklearn import datasets
from sklearn.utils import check_random_state as sk_check_random_state

import dask_ml.decomposition as dd
from dask_ml._compat import DASK_2_26_0
from dask_ml.utils import assert_estimator_equal, flip_vector_signs, svd_flip

iris = datasets.load_iris()
solver_list = ["full", "randomized", "auto", "tsqr"]
X = iris.data
n_samples, n_features = X.shape
dX = da.from_array(X, chunks=(n_samples // 2, n_features))


def test_basic():
    a = dd.PCA()
    b = sd.PCA()
    a.fit(dX)
    b.fit(X)
    assert_estimator_equal(a, b, exclude=["components_"])
    np.testing.assert_allclose(
        flip_vector_signs(a.components_, 1), flip_vector_signs(b.components_, 1)
    )


def test_pca():
    # PCA on dense arrays

    for n_comp in np.arange(X.shape[1]):
        pca = dd.PCA(n_components=n_comp, svd_solver="full")

        X_r = pca.fit(dX).transform(dX)
        np.testing.assert_equal(X_r.shape[1], n_comp)

        X_r2 = pca.fit_transform(dX)
        # TODO: Have to call compute here, since these don't pass _check_dsk
        assert_eq(X_r.compute(), X_r2.compute())

        X_r = pca.transform(dX)
        X_r2 = pca.fit_transform(dX)
        assert_array_almost_equal(X_r, X_r2)

        # Test get_covariance and get_precision
        cov = pca.get_covariance()
        precision = pca.get_precision()
        assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    # test explained_variance_ratio_ == 1 with all components
    pca = dd.PCA(svd_solver="full")
    pca.fit(dX)
    assert_almost_equal(pca.explained_variance_ratio_.sum(), 1.0, 3)


def test_pca_randomized_solver():
    # Loop excluding the 0, invalid for randomized
    for n_comp in np.arange(1, dX.shape[1]):
        pca = dd.PCA(
            n_components=n_comp,
            svd_solver="randomized",
            random_state=0,
            iterated_power=4,
        )

        X_r = pca.fit(dX).transform(dX)
        np.testing.assert_equal(X_r.shape[1], n_comp)

        X_r2 = pca.fit_transform(dX)
        # TODO: fails assert_eq
        assert_eq(X_r.compute(), X_r2.compute(), atol=1e-3)

        X_r = pca.transform(dX)
        # TODO: fails assert_eq
        assert_eq(X_r.compute(), X_r2.compute(), atol=1e-3)

        # Test get_covariance and get_precision
        cov = pca.get_covariance()
        precision = pca.get_precision()
        assert_array_almost_equal(np.dot(cov, precision), np.eye(X.shape[1]), 12)

    pca = dd.PCA(n_components=0, svd_solver="randomized", random_state=0)
    with pytest.raises(ValueError):
        pca.fit(dX)

    # Check internal state
    assert (
        pca.n_components
        == dd.PCA(n_components=0, svd_solver="randomized", random_state=0).n_components
    )
    assert (
        pca.svd_solver
        == dd.PCA(n_components=0, svd_solver="randomized", random_state=0).svd_solver
    )


def test_no_empty_slice_warning():
    if not DASK_2_26_0:
        # See https://github.com/dask/dask/pull/6591
        pytest.xfail("Dask SVD with wide arrays not supported until 2.26.0")
    # test if we avoid numpy warnings for computing over empty arrays
    n_components = 10
    n_features = n_components + 2  # anything > n_comps triggered it in 0.16
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    dX = da.from_array(X, chunks=(n_components, n_features))
    pca = dd.PCA(n_components=n_components)
    with pytest.warns(None) as w:
        pca.fit(dX)

    assert len(w) == 0


def test_whitening():
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # some low rank data with correlated features
    X = np.dot(
        rng.randn(n_samples, rank),
        np.dot(np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
    )
    # the component-wise variance of the first 50 features is 3 times the
    # mean component-wise variance of the remaining 30 features
    X[:, :50] *= 3

    assert X.shape == (n_samples, n_features)

    # the component-wise variance is thus highly varying:
    assert X.std(axis=0).std() > 43.8
    dX = da.from_array(X, chunks=(50, n_features))

    for solver, copy in product(solver_list, (True, False)):
        # whiten the data while projecting to the lower dim subspace
        X_ = dX.copy()  # make sure we keep an original across iterations.
        pca = dd.PCA(
            n_components=n_components,
            whiten=True,
            copy=copy,
            svd_solver=solver,
            random_state=0,
            iterated_power=4,
        )
        # test fit_transform
        X_whitened = pca.fit_transform(X_.copy())
        assert X_whitened.shape == (n_samples, n_components)
        # X_whitened2 = pca.transform(X_)
        # XXX: These differ for randomized.
        # assert_eq(X_whitened.compute(), X_whitened2.compute(),
        #           atol=tol, rtol=tol)

        assert_almost_equal(
            X_whitened.std(ddof=1, axis=0), np.ones(n_components), decimal=6
        )
        assert_almost_equal(X_whitened.mean(axis=0), np.zeros(n_components))

        X_ = dX.copy()
        pca = dd.PCA(
            n_components=n_components,
            whiten=False,
            copy=copy,
            svd_solver=solver,
            random_state=0,
        ).fit(X_)
        X_unwhitened = pca.transform(X_)
        assert X_unwhitened.shape == (n_samples, n_components)

        # in that case the output components still have varying variances
        assert_almost_equal(X_unwhitened.std(axis=0).std(), 74.1, 1)
        # we always center, so no test for non-centering.


# Ignore warnings from switching to more power iterations in randomized_svd
@pytest.mark.filterwarnings("ignore")
def test_explained_variance():
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80

    X = rng.randn(n_samples, n_features)
    dX = da.from_array(X, chunks=(50, n_features))

    pca = sd.PCA(n_components=2, svd_solver="full").fit(X)
    apca = dd.PCA(n_components=2, svd_solver="full", random_state=0).fit(dX)
    assert_array_almost_equal(pca.explained_variance_, apca.explained_variance_, 1)
    assert_array_almost_equal(
        pca.explained_variance_ratio_, apca.explained_variance_ratio_, 3
    )
    assert_array_almost_equal(pca.noise_variance_, apca.noise_variance_, 3)

    rpca = dd.PCA(
        n_components=2, svd_solver="randomized", random_state=42, iterated_power=1
    ).fit(dX)
    assert_array_almost_equal(pca.explained_variance_, rpca.explained_variance_, 1)
    assert_array_almost_equal(
        pca.explained_variance_ratio_, rpca.explained_variance_ratio_, 1
    )
    assert_array_almost_equal(pca.noise_variance_, rpca.noise_variance_, 1)

    # compare to empirical variances
    expected_result = np.linalg.eig(np.cov(X, rowvar=False))[0]
    expected_result = sorted(expected_result, reverse=True)[:2]

    X_pca = apca.transform(X)
    assert_array_almost_equal(apca.explained_variance_, np.var(X_pca, ddof=1, axis=0))
    assert_array_almost_equal(apca.explained_variance_, expected_result)

    X_rpca = rpca.transform(X)
    assert_array_almost_equal(
        rpca.explained_variance_, np.var(X_rpca, ddof=1, axis=0), decimal=1
    )
    assert_array_almost_equal(rpca.explained_variance_, expected_result, decimal=1)

    # Same with correlated data
    X = datasets.make_classification(
        n_samples, n_features, n_informative=n_features - 2, random_state=rng
    )[0]
    dX = da.from_array(X, chunks=(50, n_features))

    pca = sd.PCA(n_components=2).fit(X)
    rpca = dd.PCA(
        n_components=2, svd_solver="randomized", random_state=rng, iterated_power=2
    ).fit(dX)
    assert_array_almost_equal(
        pca.explained_variance_ratio_, rpca.explained_variance_ratio_, 5
    )


def test_singular_values():
    # Check that the PCA output has the correct singular values

    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80

    X = rng.randn(n_samples, n_features)
    dX = da.from_array(X, chunks=(50, n_features))

    pca = sd.PCA(n_components=2, svd_solver="full", random_state=rng).fit(X)
    apca = dd.PCA(n_components=2, svd_solver="full", random_state=rng).fit(dX)
    rpca = dd.PCA(
        n_components=2, svd_solver="randomized", random_state=rng, iterated_power=4
    ).fit(dX)
    assert_array_almost_equal(pca.singular_values_, apca.singular_values_, 12)
    assert_array_almost_equal(pca.singular_values_, rpca.singular_values_, 1)
    assert_array_almost_equal(apca.singular_values_, rpca.singular_values_, 1)

    # Compare to the Frobenius norm
    X_pca = pca.transform(X)
    X_apca = apca.transform(dX)
    X_rpca = rpca.transform(dX)
    assert_array_almost_equal(
        np.sum(pca.singular_values_ ** 2.0), np.linalg.norm(X_pca, "fro") ** 2.0, 12
    )
    assert_array_almost_equal(
        np.sum(apca.singular_values_ ** 2.0), np.linalg.norm(X_apca, "fro") ** 2.0, 9
    )
    assert_array_almost_equal(
        np.sum(rpca.singular_values_ ** 2.0), np.linalg.norm(X_rpca, "fro") ** 2.0, 0
    )

    # Compare to the 2-norms of the score vectors
    assert_array_almost_equal(
        pca.singular_values_, np.sqrt(np.sum(X_pca ** 2.0, axis=0)), 12
    )
    assert_array_almost_equal(
        apca.singular_values_, np.sqrt(np.sum(X_apca ** 2.0, axis=0)), 12
    )
    assert_array_almost_equal(
        rpca.singular_values_, np.sqrt(np.sum(X_rpca ** 2.0, axis=0)), 2
    )


def test_singular_values_wide():
    if not DASK_2_26_0:
        # See https://github.com/dask/dask/pull/6591
        pytest.xfail("Dask SVD with wide arrays not supported until 2.26.0")
    # This is split off test_singular_values, but we can't pass it ATM
    # Set the singular values and see what we get back
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 110

    X = rng.randn(n_samples, n_features)

    pca = sd.PCA(n_components=3, svd_solver="full", random_state=rng)
    apca = dd.PCA(n_components=3, svd_solver="full", random_state=rng)
    rpca = dd.PCA(n_components=3, svd_solver="randomized", random_state=rng)
    X_pca = pca.fit_transform(X)

    X_pca /= np.sqrt(np.sum(X_pca ** 2.0, axis=0))
    X_pca[:, 0] *= 3.142
    X_pca[:, 1] *= 2.718

    X_hat = np.dot(X_pca, pca.components_)
    dX_hat = da.from_array(X_hat, chunks=X_hat.shape)
    pca.fit(X_hat)
    apca.fit(dX_hat)
    rpca.fit(dX_hat)
    assert_array_almost_equal(pca.singular_values_, [3.142, 2.718, 1.0], 14)
    assert_array_almost_equal(apca.singular_values_, [3.142, 2.718, 1.0], 14)
    assert_array_almost_equal(rpca.singular_values_, [3.142, 2.718, 1.0], 14)


def test_pca_check_projection():
    # Test that the projection of data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])
    dX = da.from_array(X, chunks=(n, p))
    dXt = da.from_array(Xt, chunks=(n, p))

    for solver in solver_list:
        Yt = dd.PCA(n_components=2, svd_solver=solver).fit(dX).transform(dXt)
        Yt /= np.sqrt((Yt ** 2).sum())

        assert_almost_equal(np.abs(Yt[0][0]), 1.0, 1)


def test_pca_inverse():
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean
    dX = da.from_array(X, chunks=(n // 2, p))

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    pca = dd.PCA(n_components=2, svd_solver="full").fit(dX)
    Y = pca.transform(dX)
    Y_inverse = pca.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=3)

    # same as above with whitening (approximate reconstruction)
    for solver in solver_list:
        pca = dd.PCA(n_components=2, whiten=True, svd_solver=solver)
        pca.fit(dX)
        Y = pca.transform(dX)
        Y_inverse = pca.inverse_transform(Y)
        assert_eq(dX, Y_inverse, atol=1e-3)


def test_pca_validation():
    # Ensures that solver-specific extreme inputs for the n_components
    # parameter raise errors
    X = np.array([[0, 1, 0], [1, 0, 0]])
    X = da.from_array(X, chunks=(2, 3))
    smallest_d = 2  # The smallest dimension

    for solver in solver_list:
        # We conduct the same test on X.T so that it is invariant to axis.
        # But dask-ml needs tall and skinny
        for data in [X]:
            for n_components in [-1, 3]:

                with pytest.raises(ValueError, match="n_components"):
                    dd.PCA(n_components, svd_solver=solver).fit(data)

            if solver == "arpack":

                n_components = smallest_d
                with pytest.raises(ValueError, match="n_components"):
                    dd.PCA(n_components, svd_solver=solver).fit(data)


def test_n_components_none():
    # Ensures that n_components == None is handled correctly
    X = iris.data
    dX = da.from_array(X, chunks=X.shape)
    # We conduct the same test on X.T so that it is invariant to axis.
    # dask-ml has the extra restriction of a single block on axis 1
    # and Tall and skinny, so we skip X.T
    for data in [dX]:
        for solver in solver_list:
            pca = dd.PCA(svd_solver=solver)
            pca.fit(data)
            if solver == "arpack":
                assert pca.n_components_ == min(data.shape) - 1
            else:
                assert pca.n_components_ == min(data.shape)


def test_randomized_pca_check_projection():
    # Test that the projection by randomized PCA on dense data is correct
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])
    X = da.from_array(X, chunks=(n, p))
    dXt = da.from_array(Xt, chunks=(n, p))

    Yt = (
        dd.PCA(n_components=2, svd_solver="randomized", random_state=0)
        .fit(X)
        .transform(dXt)
    )
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_almost_equal(np.abs(Yt[0][0]), 1.0, 1)


@pytest.mark.xfail(reason="chunks")
def test_randomized_pca_check_list():
    # Test that the projection by randomized PCA on list data is correct
    X = [[1.0, 0.0], [0.0, 1.0]]
    X_transformed = (
        dd.PCA(n_components=1, svd_solver="randomized", random_state=0)
        .fit(X)
        .transform(X)
    )
    assert X_transformed.shape == (2, 1)
    assert_almost_equal(X_transformed.mean(), 0.00, 2)
    assert_almost_equal(X_transformed.std(), 0.71, 2)


def test_randomized_pca_inverse():
    # Test that randomized PCA is inversible on dense data
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean
    dX = da.from_array(X, chunks=(n, p))

    # same check that we can find the original data from the transformed signal
    # (since the data is almost of rank n_components)
    pca = dd.PCA(n_components=2, svd_solver="randomized", random_state=0).fit(dX)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=2)

    # same as above with whitening (approximate reconstruction)
    pca = dd.PCA(
        n_components=2, whiten=True, svd_solver="randomized", random_state=0
    ).fit(dX)
    Y = pca.transform(X)
    Y_inverse = pca.inverse_transform(Y)
    relative_max_delta = (np.abs(X - Y_inverse) / np.abs(X).mean()).max()
    assert relative_max_delta < 1e-5


@pytest.mark.xfail(reason="MLE")
def test_pca_dim():
    # Check automated dimensionality setting
    rng = np.random.RandomState(0)
    n, p = 100, 5
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    pca = dd.PCA(n_components="mle", svd_solver="full").fit(X)
    assert pca.n_components == "mle"
    assert pca.n_components_ == 1


def test_infer_dim_1():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = (
        rng.randn(n, p) * 0.1
        + rng.randn(n, 1) * np.array([3, 4, 5, 1, 2])
        + np.array([1, 0, 7, 4, 6])
    )
    X = da.from_array(X, chunks=(n, p))
    pca = dd.PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    # These tests rely on private imports from scikit-learn
    # spect = pca.explained_variance_
    # ll = []
    # for k in range(p):
    #     ll.append(_assess_dimension_(spect, k, n, p))
    # ll = np.array(ll)
    # assert ll[1] > ll.max() - 0.01 * n


def test_infer_dim_2():
    # TODO: explain what this is testing
    # Or at least use explicit variable names...
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    dX = da.from_array(X, chunks=(n, p))
    pca = dd.PCA(n_components=p, svd_solver="full")
    pca.fit(dX)
    # spect = pca.explained_variance_
    # assert _infer_dimension_(spect, n, p) > 1


def test_infer_dim_3():
    n, p = 100, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    X[30:40] += 2 * np.array([-1, 1, -1, 1, -1])
    X = da.from_array(X, chunks=(n, p))
    pca = dd.PCA(n_components=p, svd_solver="full")
    pca.fit(X)
    # spect = pca.explained_variance_
    # assert _infer_dimension_(spect, n, p) > 2


@pytest.mark.xfail(reason="Fractional n_components")
def test_infer_dim_by_explained_variance():
    X = da.from_array(iris.data, chunks=iris.data.shape)
    pca = dd.PCA(n_components=0.95, svd_solver="full")
    pca.fit(X)
    assert pca.n_components == 0.95
    assert pca.n_components_ == 2

    pca = dd.PCA(n_components=0.01, svd_solver="full")
    pca.fit(X)
    assert pca.n_components == 0.01
    assert pca.n_components_ == 1

    # Can't do this
    rng = np.random.RandomState(0)
    # more features than samples
    X = rng.rand(5, 20)
    pca = dd.PCA(n_components=0.5, svd_solver="full").fit(X)
    assert pca.n_components == 0.5
    assert pca.n_components_ == 2


def test_pca_score():
    # Test that probabilistic PCA scoring yields a reasonable score
    n, p = 1000, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    dX = da.from_array(X, chunks=(n // 2, p))
    for solver in solver_list:
        pca = dd.PCA(n_components=2, svd_solver=solver)
        pca.fit(dX)
        ll1 = pca.score(dX)
        h = -0.5 * np.log(2 * np.pi * np.exp(1) * 0.1 ** 2) * p
        np.testing.assert_almost_equal(ll1 / h, 1, 0)


def test_pca_score2():
    # Test that probabilistic PCA correctly separated different datasets
    n, p = 100, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    dX = da.from_array(X, chunks=(n // 2, p))
    for solver in solver_list:
        pca = dd.PCA(n_components=2, svd_solver=solver)
        pca.fit(dX)
        ll1 = pca.score(dX)
        ll2 = pca.score(rng.randn(n, p) * 0.2 + np.array([3, 4, 5]))
        assert ll1 > ll2

        # Test that it gives different scores if whiten=True
        pca = dd.PCA(n_components=2, whiten=True, svd_solver=solver)
        pca.fit(dX)
        ll2 = pca.score(dX)
        assert ll1 > ll2


def test_pca_score3():
    # Check that probabilistic PCA selects the right model
    n, p = 200, 3
    rng = np.random.RandomState(0)
    Xl = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    Xt = rng.randn(n, p) + rng.randn(n, 1) * np.array([3, 4, 5]) + np.array([1, 0, 7])
    ll = np.zeros(p)
    dXl = da.from_array(Xl, chunks=(n // 2, p))
    dXt = da.from_array(Xt, chunks=(n // 2, p))
    for k in range(p):
        pca = dd.PCA(n_components=k, svd_solver="full")
        pca.fit(dXl)
        ll[k] = pca.score(dXt)

    assert ll.argmax() == 1


def test_pca_score_with_different_solvers():
    digits = datasets.load_digits()
    X_digits = digits.data

    dX_digits = da.from_array(X_digits, chunks=X_digits.shape)

    pca_dict = {
        svd_solver: dd.PCA(
            n_components=30, svd_solver=svd_solver, random_state=0, iterated_power=3
        )
        for svd_solver in solver_list
    }

    for pca in pca_dict.values():
        pca.fit(dX_digits)
        # Sanity check for the noise_variance_. For more details see
        # https://github.com/scikit-learn/scikit-learn/issues/7568
        # https://github.com/scikit-learn/scikit-learn/issues/8541
        # https://github.com/scikit-learn/scikit-learn/issues/8544
        assert np.all((pca.explained_variance_ - pca.noise_variance_) >= 0)

    # Compare scores with different svd_solvers
    score_dict = {
        svd_solver: pca.score(dX_digits) for svd_solver, pca in pca_dict.items()
    }
    assert_almost_equal(score_dict["full"], score_dict["randomized"], decimal=3)


def test_pca_zero_noise_variance_edge_cases():
    # ensure that noise_variance_ is 0 in edge cases
    # when n_components == min(n_samples, n_features)
    n, p = 100, 3

    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 5])
    dX = da.from_array(X, chunks=(n, p))
    # arpack raises ValueError for n_components == min(n_samples,
    # n_features)
    svd_solvers = ["full", "randomized"]

    for svd_solver in svd_solvers:
        pca = dd.PCA(svd_solver=svd_solver, n_components=p)
        pca.fit(dX)
        assert pca.noise_variance_ == 0

        # Can't handle short and wide
        # pca.fit(X.T)
        # assert pca.noise_variance_ == 0


# removed test_svd_solver_auto, as we don't do that.
# removed test_deprecation_randomized_pca, as we don't do that
# removed test_pca_sparse_input: covered by test_pca_sklearn_inputs


def test_pca_bad_solver():
    X = np.random.RandomState(0).rand(5, 4)
    pca = dd.PCA(n_components=3, svd_solver="bad_argument")
    assert_raises(ValueError, pca.fit, da.from_array(X))


# def test_pca_dtype_preservation():
#     for svd_solver in solver_list:
#         yield check_pca_float_dtype_preservation, svd_solver
#         yield check_pca_int_dtype_upcast_to_double, svd_solver


@pytest.mark.parametrize(
    "svd_solver",
    [
        "full",
        pytest.param(
            "randomized", marks=pytest.mark.xfail(reason="svd_compressed promotes")
        ),
    ],
)
def test_pca_float_dtype_preservation(svd_solver):
    # Ensure that PCA does not upscale the dtype when input is float32
    X_64 = np.random.RandomState(0).rand(1000, 4).astype(np.float64)
    X_32 = X_64.astype(np.float32)

    dX_64 = da.from_array(X_64, chunks=X_64.shape)
    dX_32 = da.from_array(X_32, chunks=X_64.shape)

    pca_64 = dd.PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(dX_64)
    pca_32 = dd.PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(dX_32)

    assert pca_64.components_.dtype == np.float64
    assert pca_64.transform(dX_64).dtype == np.float64
    if DASK_2_26_0:
        # See https://github.com/dask/dask/pull/6643
        pytest.xfail("SVD dtype not preserved in dask 2.26.0")
    else:
        assert pca_32.components_.dtype == np.float32
        assert pca_32.transform(dX_32).dtype == np.float32

    assert_array_almost_equal(pca_64.components_, pca_32.components_, decimal=5)


@pytest.mark.parametrize("svd_solver", solver_list)
def test_pca_int_dtype_upcast_to_double(svd_solver):
    # Ensure that all int types will be upcast to float64
    X_i64 = np.random.RandomState(0).randint(0, 1000, (1000, 4))
    X_i64 = X_i64.astype(np.int64)
    X_i32 = X_i64.astype(np.int32)

    dX_i64 = da.from_array(X_i64, chunks=X_i64.shape)
    dX_i32 = da.from_array(X_i32, chunks=X_i32.shape)

    pca_64 = dd.PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(dX_i64)
    pca_32 = dd.PCA(n_components=3, svd_solver=svd_solver, random_state=0).fit(dX_i32)

    assert pca_64.components_.dtype == np.float64
    assert pca_32.components_.dtype == np.float64
    assert pca_64.transform(dX_i64).dtype == np.float64
    assert pca_32.transform(dX_i32).dtype == np.float64

    assert_array_almost_equal(pca_64.components_, pca_32.components_, decimal=5)


def test_fractional_n_components():
    X = da.random.uniform(size=(10, 5), chunks=(5, 5))
    pca = dd.PCA(n_components=0.5)
    with pytest.raises(NotImplementedError) as w:
        pca.fit(X)

    assert w.match("Fractional 'n_components'")


@pytest.mark.parametrize("solver", ["auto", "tsqr", "randomized", "full"])
@pytest.mark.parametrize("fn", ["fit", "fit_transform"])
def test_unknown_shapes(fn, solver):
    rng = sk_check_random_state(42)
    X = rng.uniform(-1, 1, size=(10, 3))
    df = pd.DataFrame(X)
    ddf = dask.dataframe.from_pandas(df, npartitions=2)

    pca = dd.PCA(n_components=2, svd_solver=solver)
    fit_fn = getattr(pca, fn)
    X = ddf.values
    assert np.isnan(X.shape[0])

    if solver == "auto":
        with pytest.raises(ValueError, match="Cannot automatically choose PCA solver"):
            fit_fn(X)
    else:
        X_hat = fit_fn(X)
        assert hasattr(pca, "components_")
        assert pca.n_components_ == 2
        assert pca.n_features_ == 3
        assert pca.n_samples_ == 10
        if fn == "fit_transform":
            assert np.isnan(X_hat.shape[0])
            assert X_hat.shape[1] == 2


@pytest.mark.parametrize("solver", ["randomized", "tsqr", "full"])
def test_unknown_shapes_n_components_larger_than_num_rows(solver):
    X = np.random.randn(2, 10)
    df = pd.DataFrame(X)
    ddf = dask.dataframe.from_pandas(df, npartitions=2)
    X = ddf.values
    assert np.isnan(X.shape[0])

    pca = dd.PCA(n_components=3, svd_solver=solver)
    if solver == "randomized":
        with pytest.raises(
            ValueError,
            match="n_components=3 is larger than the number of singular values",
        ) as error:
            pca.fit(X)
        assert "PCA has attributes as if n_components == 2" in str(error.value)
        assert pca.n_components_ == 2
        assert len(pca.singular_values_) == 2
        assert len(pca.components_) == 2
        assert pca.n_features_ == 10
        assert pca.n_samples_ == 2
        if solver != "randomized":
            assert pca.explained_variance_ratio_.max() == 1.0
    else:
        if DASK_2_26_0:
            # With dask 2.26.0+, tsqr will no longer throw an error related to svd_flip
            # when inputs have more columns than rows so the expected failure here
            # should come when the number of components is checked against the number
            # of singular values, i.e. this is the error that should get thrown
            with pytest.raises(
                ValueError,
                match="n_components=3 is larger than the number of singular values",
            ):
                pca.fit(X)
        else:
            # Prior to dask 2.26.0, the error being wrapped and rethrown as a
            # problem related to having too many components was actually caused
            # by there being more columns than rows in the input data
            with pytest.raises(ValueError, match="n_components is too large"):
                pca.fit(X)


@pytest.mark.parametrize("input_type", [np.array, pd.DataFrame, sp.sparse.csr_matrix])
@pytest.mark.parametrize("solver", solver_list)
def test_pca_sklearn_inputs(input_type, solver):
    Y = input_type(X)

    a = dd.PCA()
    with pytest.raises(TypeError, match="unsupported type"):
        a.fit(Y)
    with pytest.raises(TypeError, match="unsupported type"):
        a.fit_transform(Y)


@pytest.mark.parametrize("u_based", [True, False])
def test_svd_flip(u_based):
    rng = np.random.RandomState(0)
    u = rng.randn(8, 3)
    v = rng.randn(3, 10)
    u = da.from_array(u, chunks=(-1, -1))
    v = da.from_array(v, chunks=(-1, -1))
    u2, v2 = svd_flip(u, v, u_based_decision=u_based)

    def set_readonly(x):
        x.setflags(write=False)
        return x

    u = u.map_blocks(set_readonly)
    v = v.map_blocks(set_readonly)
    u, v = svd_flip(u, v, u_based_decision=u_based)
    assert_eq(u, u2)
    assert_eq(v, v2)
