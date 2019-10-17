import dask
import dask.array as da
import dask.multiprocessing
import numpy as np
import pytest
from dask import persist

from dask_ml.linear_model.algorithms import (
    admm,
    gradient_descent,
    lbfgs,
    newton,
    proximal_grad,
)
from dask_ml.linear_model.families import Logistic, Normal, Poisson
from dask_ml.linear_model.regularizers import Regularizer
from dask_ml.linear_model.utils import make_y, sigmoid


def add_l1(f, lam):
    def wrapped(beta, X, y):
        return f(beta, X, y) + lam * (np.abs(beta)).sum()

    return wrapped


def make_intercept_data(N, p, seed=20009):
    """Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on."""

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


@pytest.mark.parametrize("opt", [lbfgs, newton, gradient_descent])
@pytest.mark.parametrize(
    "N, p, seed,", [(100, 2, 20009), (250, 12, 90210), (95, 6, 70605)]
)
def test_methods(N, p, seed, opt):
    X, y = make_intercept_data(N, p, seed=seed)
    coefs, _ = opt(X, y)
    p = sigmoid(X.dot(coefs).compute())

    y_sum = y.compute().sum()
    p_sum = p.sum()
    assert np.isclose(y_sum, p_sum, atol=1e-1)


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (newton, {"tol": 1e-5, "max_iter": 50}),
        (lbfgs, {"tol": 1e-8, "max_iter": 100}),
        (gradient_descent, {"tol": 1e-7, "max_iter": 100}),
    ],
)
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("nchunks", [1, 10])
@pytest.mark.parametrize("family", [Logistic, Normal, Poisson])
def test_basic_unreg_descent(func, kwargs, N, nchunks, family):
    beta = np.random.normal(size=2)
    M = len(beta)
    X = da.random.random((N, M), chunks=(N // nchunks, M))
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = persist(X, y)

    result, n_iter = func(X, y, family=family, **kwargs)
    test_vec = np.random.normal(size=2)

    opt = family.pointwise_loss(result, X, y).compute()
    test_val = family.pointwise_loss(test_vec, X, y).compute()

    max_iter = kwargs["max_iter"]
    assert n_iter > 0 and n_iter <= max_iter
    assert opt < test_val


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (admm, {"abstol": 1e-4, "max_iter": 250}),
        (proximal_grad, {"tol": 1e-7, "max_iter": 100}),
    ],
)
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("nchunks", [1, 10])
@pytest.mark.parametrize("family", [Logistic, Normal, Poisson])
@pytest.mark.parametrize("lam", [0.01, 1.2, 4.05])
@pytest.mark.parametrize("reg", [r() for r in Regularizer.__subclasses__()])
def test_basic_reg_descent(func, kwargs, N, nchunks, family, lam, reg):
    beta = np.random.normal(size=2)
    M = len(beta)
    X = da.random.random((N, M), chunks=(N // nchunks, M))
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = persist(X, y)

    result, n_iter = func(X, y, family=family, lamduh=lam, regularizer=reg, **kwargs)
    test_vec = np.random.normal(size=2)

    f = reg.add_reg_f(family.pointwise_loss, lam)

    opt = f(result, X, y).compute()
    test_val = f(test_vec, X, y).compute()

    max_iter = kwargs["max_iter"]
    assert n_iter > 0 and n_iter <= max_iter
    assert opt < test_val


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (admm, {"max_iter": 2}),
        (proximal_grad, {"max_iter": 2}),
        (newton, {"max_iter": 2}),
        (gradient_descent, {"max_iter": 2}),
    ],
)
@pytest.mark.parametrize("scheduler", ["synchronous", "threading", "multiprocessing"])
def test_determinism(func, kwargs, scheduler):
    X, y = make_intercept_data(1000, 10)

    with dask.config.set(scheduler=scheduler):
        a, n_iter_a = func(X, y, **kwargs)
        b, n_iter_b = func(X, y, **kwargs)

    max_iter = kwargs["max_iter"]
    assert n_iter_a > 0 and n_iter_a <= max_iter
    assert n_iter_b > 0 and n_iter_b <= max_iter
    assert (a == b).all()


try:
    from distributed import Client
    from distributed.utils_test import cluster, loop  # noqa
except ImportError:
    pass
else:

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (admm, {"max_iter": 2}),
            (proximal_grad, {"max_iter": 2}),
            (newton, {"max_iter": 2}),
            (gradient_descent, {"max_iter": 2}),
        ],
    )
    def test_determinism_distributed(func, kwargs, loop):
        with cluster() as (s, [a, b]):
            with Client(s["address"], loop=loop) as c:
                X, y = make_intercept_data(1000, 10)

                a, n_iter_a = func(X, y, **kwargs)
                b, n_iter_b = func(X, y, **kwargs)

                max_iter = kwargs["max_iter"]
                assert n_iter_a > 0 and n_iter_a <= max_iter
                assert n_iter_b > 0 and n_iter_b <= max_iter
                assert (a == b).all()

    def broadcast_lbfgs_weight():
        with cluster() as (s, [a, b]):
            with Client(s["address"], loop=loop) as c:
                X, y = make_intercept_data(1000, 10)
                coefs = lbfgs(X, y, dask_distributed_client=c)
                p = sigmoid(X.dot(coefs).compute())

                y_sum = y.compute().sum()
                p_sum = p.sum()
                assert np.isclose(y_sum, p_sum, atol=1e-1)
