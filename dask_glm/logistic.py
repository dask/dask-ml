from __future__ import absolute_import, division, print_function

from dask import delayed, persist, compute
import numpy as np
import dask.array as da
from scipy.optimize import fmin_l_bfgs_b

from dask_glm.utils import dot, exp, log1p, absolute, sign


try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def _(func):
            return func
        return _


def bfgs(X, y, max_iter=500, tol=1e-14):
    '''Simple implementation of BFGS.'''

    n, p = X.shape
    y = y.squeeze()

    recalcRate = 10
    stepSize = 1.0
    armijoMult = 1e-4
    backtrackMult = 0.5
    stepGrowth = 1.25

    beta = np.zeros(p)
    Hk = np.eye(p)
    for k in range(max_iter):

        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            eXbeta = exp(Xbeta)
            func = log1p(eXbeta).sum() - dot(y, Xbeta)

        e1 = eXbeta + 1.0
        gradient = dot(X.T, eXbeta / e1 - y)  # implicit numpy -> dask conversion

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
        stepSize, _, _, func = delayed(compute_stepsize, nout=4)(beta,
                                                                 step,
                                                                 Xbeta,
                                                                 Xstep,
                                                                 y,
                                                                 func,
                                                                 backtrackMult=backtrackMult,
                                                                 armijoMult=armijoMult,
                                                                 stepSize=stepSize)

        beta, stepSize, Xbeta, gradient, lf, func, step, Xstep = persist(
            beta, stepSize, Xbeta, gradient, lf, func, step, Xstep)

        stepSize, lf, func, step = compute(stepSize, lf, func, step)

        beta = beta - stepSize * step  # tiny bit of repeat work here to avoid communication
        Xbeta = Xbeta - stepSize * Xstep

        if stepSize == 0:
            print('No more progress')
            break

        # necessary for gradient computation
        eXbeta = exp(Xbeta)

        yk = -gradient
        sk = -stepSize * step
        stepSize *= stepGrowth

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
    y_local = y.compute()

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
    beta = np.zeros(p)  # always init to zeros?
    Xbeta = dot(X, beta)

    iter_count = 0
    converged = False

    while not converged:
        beta_old = beta

        # should this use map_blocks()?
        p = sigmoid(Xbeta)
        hessian = dot(p * (1 - p) * X.T, X)
        grad = dot(X.T, p - y)

        hessian, grad = da.compute(hessian, grad)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, _, _, _ = np.linalg.lstsq(hessian, grad)
        beta = (beta_old - step)

        iter_count += 1

        # should change this criterion
        coef_change = np.absolute(beta_old - beta)
        converged = (
            (not np.any(coef_change > tol)) or (iter_count > max_iter))

        if not converged:
            Xbeta = dot(X, beta)  # numpy -> dask converstion of beta

    return beta


def proximal_grad(X, y, reg='l2', lamduh=0.1, max_steps=100, tol=1e-8):
    def l2(x, t):
        return 1 / (1 + lamduh * t) * x

    def l1(x, t):
        return (absolute(x) > lamduh * t) * (x - sign(x) * lamduh * t)

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
            eXbeta = exp(Xbeta)
            func = log1p(eXbeta).sum() - y.dot(Xbeta)
        e1 = eXbeta + 1.0
        gradient = X.T.dot(eXbeta / e1 - y)

        Xbeta, eXbeta, func, gradient = persist(
            Xbeta, eXbeta, func, gradient)

        obeta = beta
        oXbeta = Xbeta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = prox_map[reg](obeta - stepSize * gradient, stepSize)
            step = obeta - beta
            Xbeta = X.dot(beta)

            overflow = (Xbeta < 700).all()
            overflow, Xbeta, beta = persist(overflow, Xbeta, beta)
            overflow = overflow.compute()

            # This prevents overflow
            if overflow:
                eXbeta = exp(Xbeta)
                func = log1p(eXbeta).sum() - dot(y, Xbeta)
                eXbeta, func = persist(eXbeta, func)
                func = func.compute()
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


def admm(X, y, lamduh=0.1, rho=1, over_relax=1,
         max_iter=100, abstol=1e-4, reltol=1e-2):

    nchunks = X.npartitions
    (n, p) = X.shape
    XD = X.to_delayed().flatten().tolist()
    yD = y.to_delayed().flatten().tolist()

    z = np.zeros(p)
    u = np.array([np.zeros(p) for i in range(nchunks)])
    betas = np.array([np.zeros(p) for i in range(nchunks)])

    for k in range(max_iter):

        # x-update step
        new_betas = [delayed(local_update)(xx, yy, bb, z, uu, rho) for
                     xx, yy, bb, uu in zip(XD, yD, betas, u)]
        new_betas = np.array(da.compute(*new_betas))

        beta_hat = over_relax * new_betas + (1 - over_relax) * z

        #  z-update step
        zold = z.copy()
        ztilde = np.mean(beta_hat + np.array(u), axis=0)
        z = shrinkage(ztilde, lamduh / (rho * nchunks))

        # u-update step
        u += beta_hat - z

        # check for convergence
        primal_res = np.linalg.norm(new_betas - z)
        dual_res = np.linalg.norm(rho * (z - zold))

        eps_pri = np.sqrt(p * nchunks) * abstol + nchunks * reltol * np.maximum(
            np.linalg.norm(new_betas), np.linalg.norm(z))
        eps_dual = np.sqrt(p * nchunks) * abstol + \
            nchunks * reltol * np.linalg.norm(rho * u)

        if primal_res < eps_pri and dual_res < eps_dual:
            print("Converged!", k)
            break

    return z


# TODO: Dask+Numba JIT
def sigmoid(x):
    return 1 / (1 + exp(-x))


def logistic_loss(beta, X, y):
    '''Logistic Loss, evaluated point-wise.'''
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    eXbeta = np.exp(Xbeta)
    return np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)


def proximal_logistic_loss(beta, X, y, z, u, rho):
    return logistic_loss(beta, X, y) + (rho / 2) * np.dot(beta - z + u,
                                                          beta - z + u)


def logistic_gradient(beta, X, y):
    '''Logistic gradient, evaluated point-wise.'''
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    p = sigmoid(Xbeta)
    return X.T.dot(p - y)


def proximal_logistic_gradient(beta, X, y, z, u, rho):
    return logistic_gradient(beta, X, y) + rho * (beta - z + u)


def local_update(X, y, beta, z, u, rho, fprime=proximal_logistic_gradient,
                 f=proximal_logistic_loss,
                 solver=fmin_l_bfgs_b):
    beta = beta.ravel()
    u = u.ravel()
    z = z.ravel()
    solver_args = (X, y, z, u, rho)
    beta, f, d = solver(f, beta, fprime=fprime, args=solver_args, pgtol=1e-10,
                        maxiter=200,
                        maxfun=250, factr=1e-30)

    return beta


def shrinkage(x, kappa):
    z = np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)
    return z
