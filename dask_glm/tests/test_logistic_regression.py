from mock import Mock

import dask.array as da
import numpy as np
import pytest
from scipy.optimize import fmin_l_bfgs_b

from dask_glm.logistic import logistic_regression, local_update, \
    proximal_logistic_loss, proximal_logistic_gradient


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@pytest.fixture
def Xy(N=1000):
    beta_len = 2
    X = np.random.multivariate_normal(np.zeros(beta_len), np.eye(beta_len), N)
    y = sigmoid(X.dot(np.array([[1.5, -3]]).T)) + .001 * np.random.normal(
        size=(N, 1))
    return da.from_array(X, chunks=(N / 5, 2)), da.from_array(y, chunks=(
        N / 5, 1))


def test_logistic_regression_with_large_l1_regularization_penalty(Xy):
    alpha = 10
    rho = 1
    over_relaxation = 1
    X, y = Xy
    coeff = logistic_regression(X, y, alpha, rho, over_relaxation)
    np.testing.assert_almost_equal(coeff, np.array([0, 0]))


def test_logistic_regression_with_small_l1_regularization_penalty(Xy):
    alpha = 1e-3
    rho = 1
    over_relaxation = 1
    X, y = Xy
    coef = logistic_regression(X, y, alpha, rho, over_relaxation)

    w0, w1 = coef[0], coef[1]
    assert abs(w0 - 1.5) < 2
    assert abs(w1 + 3) < 2


def test_local_update():
    w = np.array([1, 1]).reshape(2, 1)
    X = np.array([1.5, 3])
    y = sigmoid(X)
    u = np.array([0.8, 0.8])
    z = np.array([1.2, 1.2])
    rho = 1
    coef = local_update(y, X, w, z, u, rho, f=proximal_logistic_loss,
                        fprime=proximal_logistic_gradient)

    np.testing.assert_almost_equal(coef,
                                   np.array([0.577606, 0.577606]).reshape(2,
                                                                          1))


# def test_apply_local_updates_on_a_dask_array(Xy):
#     X, y = Xy
#     w = da.from_array(np.zeros([2, 5]), chunks=(2, 1))
#     u = da.from_array(np.zeros([2, 5]), chunks=(2, 1))
#     z = np.array([1.2, 1.2])
#     rho = 1
#
#     coefs = apply_local_update(X, y, w, z, rho, u).compute()
#     assert coefs.shape == (10, 1)


def test_local_update_calls_solver_correctly():
    X, y = 1, 2
    loss_func = Mock(spec=proximal_logistic_loss)
    grad_func = Mock(spec=proximal_logistic_gradient)
    solver = Mock(spec=fmin_l_bfgs_b)
    solver.return_value = (np.array([1, 2]), None, None)
    w = Mock()
    u = Mock()
    z = Mock()
    rho = 1

    coef = local_update(y, X, w, z, u, rho, fprime=grad_func, f=loss_func,
                        solver=solver)

    solver.assert_called_once_with(loss_func, w.ravel(), fprime=grad_func,
                                   args=(X, y, z.ravel(), u.ravel(), rho),
                                   pgtol=1e-10, maxiter=200, maxfun=250,
                                   factr=1e-30)

    np.testing.assert_array_equal(coef, np.array([1, 2]).reshape(2, 1))


def test_proximal_logistic_loss():
    w = np.array([1, 1])
    X = np.array([.5, 0.9])
    y = sigmoid(X)
    u = np.array([0.8, 0.8])
    z = np.array([1.2, 1.2])
    loss = proximal_logistic_loss(w, X, y, z, u, 5)
    assert round(loss, 4) == 4.2640


def test_proximal_logistic_gradient():
    w = np.array([1, 1])
    X = np.array([.5, 0.9])
    y = sigmoid(X)
    u = np.array([0.8, 0.8])
    z = np.array([1.2, 1.2])
    loss = proximal_logistic_gradient(w, X, y, z, u, 5)
    np.testing.assert_almost_equal(loss, np.array([5.73552991, 5.73552991]))
