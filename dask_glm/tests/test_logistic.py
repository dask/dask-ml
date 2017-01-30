import pytest

from dask import persist
import numpy as np
import dask.array as da

from dask_glm.logistic import newton, bfgs, proximal_grad, gradient_descent
from dask_glm.utils import make_y


@pytest.mark.parametrize('func,kwargs', [
    (newton, {'tol': 1e-5}),
    (bfgs, {'tol': 1e-8}),
    (gradient_descent, {'tol': 1e-7}),
    (proximal_grad, {'tol': 1e-6, 'reg': 'l1', 'lamduh': 0.001}),
    (proximal_grad, {'tol': 1e-7, 'reg': 'l2', 'lamduh': 0.001}),
])
@pytest.mark.parametrize('N', [10000, 100000])
@pytest.mark.parametrize('nchunks', [1, 10])
@pytest.mark.parametrize('beta', [[-1.5, 3]])
def test_basic(func, kwargs, N, beta, nchunks):
    M = len(beta)

    X = da.random.random((N, M), chunks=(N // nchunks, M))
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = persist(X, y)

    result = func(X, y, **kwargs)

    assert np.allclose(result, beta, rtol=2e-1)
