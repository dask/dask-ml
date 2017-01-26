from __future__ import absolute_import, division, print_function

from dask import delayed, persist, compute
import numpy as np
import dask.array as da
from numba import jit
from scipy.optimize import fmin_l_bfgs_b

from dask_glm.utils import dot, exp, log1p


def bfgs(X, y, max_iter=50, tol=1e-14):
    '''Simple implementation of BFGS.'''

    n, p = X.shape

    y_local = da.compute(y)[0]

    recalcRate = 10
    stepSize = 1.0
    armijoMult = 1e-4
    backtrackMult = 0.1

    beta = np.zeros(p)
    Hk = np.eye(p)
    sk = None
    for k in range(max_iter):

        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            eXbeta = exp(Xbeta)
            func = log1p(eXbeta).sum() - y.dot(Xbeta)

        e1 = eXbeta + 1.0
        gradient = X.T.dot(
            eXbeta / e1 - y)  # implicit numpy -> dask conversion

        if k:
            yk = yk + gradient  # TODO: gradient is dasky and yk is numpy-y
            rhok = 1 / yk.dot(sk)
            adj = np.eye(p) - rhok * dot(sk, yk.T)
            Hk = dot(adj, dot(Hk, adj.T)) + rhok * dot(sk, sk.T)

        step = dot(Hk, gradient)
        steplen = dot(step, gradient)
        Xstep = dot(X, step)

        # backtracking line search
        lf = func
        old_Xbeta = Xbeta
        stepSize, beta, Xbeta, func = delayed(compute_stepsize, nout=4)(beta,
                                                                        step,
                                                                        Xbeta,
                                                                        Xstep,
                                                                        y,
                                                                        func,
                                                                        backtrackMult=backtrackMult,
                                                                        armijoMult=armijoMult,
                                                                        stepSize=stepSize)

        beta, Xstep, stepSize, Xbeta, gradient, lf, func, step = persist(
            beta, Xstep, stepSize, Xbeta, gradient, lf, func, step)

        Xbeta = da.from_delayed(Xbeta, shape=old_Xbeta.shape,
                                dtype=old_Xbeta.dtype)

        stepSize, lf, func = compute(stepSize, lf, func)

        if stepSize == 0:
            print('No more progress')
            break

        # necessary for gradient computation
        eXbeta = exp(Xbeta)

        yk = -gradient
        sk = -stepSize * step
        stepSize = 1.0

        if stepSize == 0:
            print('No more progress')

        df = lf - func
        df /= max(func, lf)
        if df < tol:
            print('Converged')
            break

    return beta


@jit(nogil=True)
def loglike(Xbeta, y):
    #        # This prevents overflow
    #        if np.all(Xbeta < 700):
    eXbeta = np.exp(Xbeta)
    return np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)


@jit(nogil=True)
def compute_stepsize(beta, step, Xbeta, Xstep, y, curr_val, stepSize=1.0,
                     armijoMult=0.1, backtrackMult=0.1):
    obeta, oXbeta = beta, Xbeta
    steplen = (step ** 2).sum()
    lf = curr_val
    func = 0
    for ii in range(100):
        beta = obeta - stepSize * step
        if ii and np.array_equal(beta, obeta):
            stepSize = 0
            break
        Xbeta = oXbeta - stepSize * Xstep

        func = loglike(Xbeta, y)
        df = lf - func
        if df >= armijoMult * stepSize * steplen:
            break
        stepSize *= backtrackMult

    return stepSize, beta, Xbeta, func


def gradient_descent(X, y, max_steps=100, tol=1e-14):
    '''Michael Grant's implementation of Gradient Descent.'''

    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)
    y_local = y.compute()  # is this different from da.compute()[0]??

    for k in range(max_steps):
        # how necessary is this recalculation?
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            eXbeta = da.exp(Xbeta)
            func = da.log1p(eXbeta).sum() - y.dot(Xbeta)

        e1 = eXbeta + 1.0
        gradient = X.T.dot(eXbeta / e1 - y)
        Xgradient = X.dot(gradient)

        Xbeta, eXbeta, func, gradient, Xgradient = da.compute(
            Xbeta, eXbeta, func, gradient, Xgradient)

        # backtracking line search
        lf = func
        stepSize, beta, Xbeta, func = compute_stepsize(beta, gradient,
                                                       Xbeta, Xgradient,
                                                       y_local, func,
                                                       **{
                                                           'backtrackMult': backtrackMult,
                                                           'armijoMult': armijoMult,
                                                           'stepSize': stepSize})
        if stepSize == 0:
            print('No more progress')
            break

        # necessary for gradient computation
        eXbeta = exp(Xbeta)

        df = lf - func
        df /= max(func, lf)

        if df < tol:
            print('Converged')
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta


def newton(X, y, max_iter=50, tol=1e-8):
    '''Newtons Method for Logistic Regression.'''

    n, p = X.shape
    beta = np.zeros(p)
    Xbeta = X.dot(beta)

    iter_count = 0
    converged = False

    while not converged:
        beta_old = beta

        # should this use map_blocks()?
        p = sigmoid(Xbeta)
        hessian = dot(p * (1 - p) * X.T, X)
        grad = X.T.dot(p - y)

        hessian, grad = da.compute(hessian, grad)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, _ = np.linalg.lstsq(hessian, grad)
        beta = (beta_old - step)

        iter_count += 1

        # should change this criterion
        coef_change = np.absolute(beta_old - beta)
        converged = (
            (not np.any(coef_change > tol)) or (iter_count > max_iter))

        if not converged:
            Xbeta = X.dot(beta)

    return beta


def proximal_grad(X, y, reg='l2', lamduh=0.1, max_steps=100, tol=1e-8):
    def l2(x, t):
        return 1 / (1 + lamduh * t) * x

    def l1(x, t):
        return (np.absolute(x) > lamduh * t) * (x - np.sign(x) * lamduh * t)

    def identity(x, t):
        return x

    prox_map = {'l1': l1, 'l2': l2, None: identity}
    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)

    print('#       -f        |df/f|    |dx/x|    step')
    print('----------------------------------------------')
    for k in range(max_steps):
        # Compute the gradient
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            eXbeta = da.exp(Xbeta)
            func = da.log1p(eXbeta).sum() - y.dot(Xbeta)
        e1 = eXbeta + 1.0
        gradient = X.T.dot(eXbeta / e1 - y)

        Xbeta, eXbeta, func, gradient = da.compute(
            Xbeta, eXbeta, func, gradient)

        obeta = beta
        oXbeta = Xbeta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = prox_map[reg](obeta - stepSize * gradient, stepSize)
            step = obeta - beta
            Xbeta = X.dot(beta).compute()  # ugh

            # This prevents overflow
            if np.all(Xbeta < 700):
                eXbeta = np.exp(Xbeta)
                func = np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)
                df = lf - func
                if df > 0:
                    break
            stepSize *= backtrackMult
        if stepSize == 0:
            print('No more progress')
            break
        df /= max(func, lf)
        db = 0
        print('%2d  %.6e %9.2e  %.2e  %.1e' % (k + 1, func, df, db, stepSize))
        if df < tol:
            print('Converged')
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta


def logistic_regression(X, y, alpha, rho, over_relaxation):
    N = 5
    (m, n) = X.shape

    z = np.zeros([n, N])
    a = np.zeros([n, N])
    beta_old = da.from_array(np.zeros([n, N]), chunks=(2, 1))
    beta = np.zeros((2, 5))
    u = da.from_array(a, chunks=(2, 1))
    MAX_ITER = 100
    ABSTOL = 1e-4
    RELTOL = 1e-2

    for k in range(MAX_ITER):
        beta_x = y.map_blocks(local_update, X, beta_old.T, z[:, 0], u.T, rho,
                              chunks=(5, 2), dtype=float).compute()
        beta[0, :] = beta_x[0::2].ravel()
        beta[1, :] = beta_x[1::2].ravel()
        beta_hat = over_relaxation * beta + (1 - over_relaxation) * z
        zold = z.copy()
        ztilde = np.mean(beta_hat + a, 1)

        z = shrinkage(ztilde, n * alpha / rho)

        z = np.tile(z.transpose(), [N, 1]).transpose()
        a += rho * (beta_hat - z)
        u = da.from_array(a, chunks=(2, 1))
        beta_old = da.from_array(beta, chunks=(2, 1))
        r_norm = np.linalg.norm(beta - z)
        s_norm = np.linalg.norm(-1 * rho * (z - zold))
        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * np.maximum(
            np.linalg.norm(beta), np.linalg.norm(-z))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(rho * a)

        if r_norm < eps_pri and s_norm < eps_dual:
            print("Converged!", k)
            break
    return z.mean(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_loss(w, X, y):
    y = y.ravel()
    z = X.dot(w)
    yz = y * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum()
    return out


def proximal_logistic_loss(w, X, y, z, u, rho):
    return logistic_loss(w, X, y) + rho * np.dot(w - z + u, w - z + u)


def logistic_gradient(w, X, y):
    z = X.dot(w)
    y = y.ravel()
    z = sigmoid(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0)
    return grad * np.ones(w.shape)


def proximal_logistic_gradient(w, X, y, z, u, rho):
    return logistic_gradient(w, X, y) + 2 * rho * (w - z + u)


def local_update(y, X, w, z, u, rho, fprime=proximal_logistic_gradient,
                 f=proximal_logistic_loss,
                 solver=fmin_l_bfgs_b):
    w = w.ravel()
    u = u.ravel()
    z = z.ravel()
    solver_args = (X, y, z, u, rho)
    w, f, d = solver(f, w, fprime=fprime, args=solver_args, pgtol=1e-10,
                     maxiter=200,
                     maxfun=250, factr=1e-30)
    return w.reshape(2, 1)


# def apply_local_update(X, y, w, z, rho, u):
#     return y.map_blocks(local_update, X, w.T, z, u.T, rho, chunks=(5, 2))


def shrinkage(x, kappa):
    z = np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)
    return z
