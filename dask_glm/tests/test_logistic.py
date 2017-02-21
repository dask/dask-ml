import pytest

from dask import persist
import numpy as np
import dask.array as da

from dask_glm.algorithms import (newton, bfgs, proximal_grad,
                                 gradient_descent, admm)
from dask_glm.utils import sigmoid, make_y


def make_data(N, p, seed=20009):
    '''Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on.'''

    # set the seeds
    da.random.seed(seed)
    np.random.seed(seed)

    X = np.random.random((N, p + 1))
    col_sums = X.sum(axis=0)
    X = X / col_sums[None, :]
    X[:, p] = 1
    X = da.from_array(X, chunks=(N / 5, p + 1))
    y = make_y(X, beta=np.random.random(p + 1))

    return X, y


@pytest.mark.parametrize('opt',
                         [pytest.mark.xfail(bfgs, reason='''
                          Early algorithm termination for unknown reason'''),
                          newton, gradient_descent])
@pytest.mark.parametrize('N, p, seed',
                         [(100, 2, 20009),
                          (250, 12, 90210),
                          (95, 6, 70605)])
def test_methods(N, p, seed, opt):
    X, y = make_data(N, p, seed=seed)
    coefs = opt(X, y)
    p = sigmoid(X.dot(coefs).compute())

    y_sum = y.compute().sum()
    p_sum = p.sum()
    assert np.isclose(y.compute().sum(), p.sum(), atol=1e-1)


@pytest.mark.parametrize('func,kwargs', [
    (newton, {'tol': 1e-5}),
    (bfgs, {'tol': 1e-8}),
    (gradient_descent, {'tol': 1e-7}),
    (proximal_grad, {'tol': 1e-6, 'reg': 'l1', 'lamduh': 0.001}),
    (proximal_grad, {'tol': 1e-7, 'reg': 'l2', 'lamduh': 0.001}),
    (admm, {}),
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
