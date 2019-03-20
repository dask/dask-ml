import pytest

from dask import persist
import dask.array as da
import numpy as np

from dask_glm.algorithms import admm, local_update
from dask_glm.families import Logistic, Normal
from dask_glm.regularizers import L1
from dask_glm.utils import make_y


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('beta',
                         [np.array([-1.5, 3]),
                          np.array([35, 2, 0, -3.2]),
                          np.array([-1e-2, 1e-4, 1.0, 2e-3, -1.2])])
@pytest.mark.parametrize('family', [Logistic, Normal])
def test_local_update(N, beta, family):
    M = beta.shape[0]
    X = np.random.random((N, M))
    y = np.random.random(N) > 0.4
    u = np.zeros(M)
    z = np.random.random(M)
    rho = 1e7

    def create_local_gradient(func):
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + rho * (beta - z + u)
        return wrapped

    def create_local_f(func):
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + (rho / 2) * np.dot(beta - z + u,
                                                         beta - z + u)
        return wrapped

    f = create_local_f(family.pointwise_loss)
    fprime = create_local_gradient(family.pointwise_gradient)

    result = local_update(X, y, beta, z, u, rho, f=f, fprime=fprime)

    assert np.allclose(result, z, atol=2e-3)


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('nchunks', [5, 10])
@pytest.mark.parametrize('p', [1, 5, 10])
def test_admm_with_large_lamduh(N, p, nchunks):
    max_iter = 500
    X = da.random.random((N, p), chunks=(N // nchunks, p))
    beta = np.random.random(p)
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = persist(X, y)
    z, n_iter = admm(X, y, regularizer=L1(), lamduh=1e5, rho=20, max_iter=max_iter)

    assert np.allclose(z, np.zeros(p), atol=1e-4)
    assert n_iter > 0 and n_iter <= max_iter
