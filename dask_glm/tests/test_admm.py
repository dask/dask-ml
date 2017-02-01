import pytest

from dask import persist
import dask.array as da
import numpy as np

from dask_glm.logistic import admm, local_update
from dask_glm.utils import make_y


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('beta',
                         [np.array([-1.5, 3]),
                          np.array([35, 2, 0, -3.2]),
                          np.array([-1e-2, 1e-4, 1.0, 2e-3, -1.2])])
def test_local_update(N, beta):
    M = beta.shape[0]
    X = np.random.random((N, M))
    y = np.random.random(N) > 0.4
    u = np.zeros(M)
    z = np.random.random(M)
    rho = 1e6

    result = local_update(X, y, beta, z, u, rho)

    assert np.allclose(result, z, atol=2e-3)


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('nchunks', [5, 10])
@pytest.mark.parametrize('p', [1, 5, 10])
def test_admm_with_large_lamduh(N, p, nchunks):
    X = da.random.random((N, p), chunks=(N // nchunks, p))
    beta = np.random.random(p)
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = persist(X, y)
    z = admm(X, y, lamduh=1e4, rho=20, max_iter=500)

    assert np.allclose(z, np.zeros(p), atol=1e-4)
