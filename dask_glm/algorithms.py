from __future__ import absolute_import, division, print_function

from dask import delayed, persist, compute
import functools
import numpy as np
import dask.array as da
from scipy.optimize import fmin_l_bfgs_b


from dask_glm.utils import dot, exp, log1p
from dask_glm.families import Logistic
from dask_glm.regularizers import L1


def compute_stepsize_dask(beta, step, Xbeta, Xstep, y, curr_val,
                          family=Logistic, stepSize=1.0,
                          armijoMult=0.1, backtrackMult=0.1):

    loglike = family.loglike
    beta, step, Xbeta, Xstep, y, curr_val = persist(beta, step, Xbeta, Xstep, y, curr_val)
    obeta, oXbeta = beta, Xbeta
    (step,) = compute(step)
    steplen = (step ** 2).sum()
    lf = curr_val
    func = 0
    for ii in range(100):
        beta = obeta - stepSize * step
        if ii and (beta == obeta).all():
            stepSize = 0
            break

        Xbeta = oXbeta - stepSize * Xstep
        func = loglike(Xbeta, y)
        Xbeta, func = persist(Xbeta, func)

        df = lf - compute(func)[0]
        if df >= armijoMult * stepSize * steplen:
            break
        stepSize *= backtrackMult

    return stepSize, beta, Xbeta, func


def gradient_descent(X, y, max_steps=100, tol=1e-14, family=Logistic):
    '''Michael Grant's implementation of Gradient Descent.'''

    loglike, gradient = family.loglike, family.gradient
    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)

    for k in range(max_steps):
        # how necessary is this recalculation?
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            func = loglike(Xbeta, y)

        grad = gradient(Xbeta, X, y)
        Xgradient = X.dot(grad)

        # backtracking line search
        lf = func
        stepSize, _, _, func = compute_stepsize_dask(beta, grad,
                                                     Xbeta, Xgradient,
                                                     y, func, family=family,
                                                     backtrackMult=backtrackMult,
                                                     armijoMult=armijoMult,
                                                     stepSize=stepSize)

        beta, stepSize, Xbeta, lf, func, grad, Xgradient = persist(
            beta, stepSize, Xbeta, lf, func, grad, Xgradient)

        stepSize, lf, func, grad = compute(stepSize, lf, func, grad)

        beta = beta - stepSize * grad  # tiny bit of repeat work here to avoid communication
        Xbeta = Xbeta - stepSize * Xgradient

        if stepSize == 0:
            print('No more progress')
            break

        df = lf - func
        df /= max(func, lf)

        if df < tol:
            print('Converged')
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta


def newton(X, y, max_steps=50, tol=1e-8, family=Logistic):
    '''Newtons Method for Logistic Regression.'''

    gradient, hessian = family.gradient, family.hessian
    n, p = X.shape
    beta = np.zeros(p)  # always init to zeros?
    Xbeta = dot(X, beta)

    iter_count = 0
    converged = False

    while not converged:
        beta_old = beta

        # should this use map_blocks()?
        hess = hessian(Xbeta, X)
        grad = gradient(Xbeta, X, y)

        hess, grad = da.compute(hess, grad)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, _, _, _ = np.linalg.lstsq(hess, grad)
        beta = (beta_old - step)

        iter_count += 1

        # should change this criterion
        coef_change = np.absolute(beta_old - beta)
        converged = (
            (not np.any(coef_change > tol)) or (iter_count > max_steps))

        if not converged:
            Xbeta = dot(X, beta)  # numpy -> dask converstion of beta

    return beta


def admm(X, y, reg=L1, lamduh=0.1, rho=1, over_relax=1,
         max_steps=100, abstol=1e-4, reltol=1e-2, family=Logistic):

    pointwise_loss = family.pointwise_loss
    pointwise_gradient = family.pointwise_gradient

    def create_local_gradient(func):
        @functools.wraps(func)
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + rho * (beta - z + u)
        return wrapped

    def create_local_f(func):
        @functools.wraps(func)
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + (rho / 2) * np.dot(beta - z + u,
                                                         beta - z + u)
        return wrapped

    f = create_local_f(pointwise_loss)
    fprime = create_local_gradient(pointwise_gradient)

    nchunks = X.npartitions
    (n, p) = X.shape
    XD = X.to_delayed().flatten().tolist()
    yD = y.to_delayed().flatten().tolist()

    z = np.zeros(p)
    u = np.array([np.zeros(p) for i in range(nchunks)])
    betas = np.array([np.zeros(p) for i in range(nchunks)])

    for k in range(max_steps):

        # x-update step
        new_betas = [delayed(local_update)(xx, yy, bb, z, uu, rho, f=f,
                                           fprime=fprime) for
                     xx, yy, bb, uu in zip(XD, yD, betas, u)]
        new_betas = np.array(da.compute(*new_betas))

        beta_hat = over_relax * new_betas + (1 - over_relax) * z

        #  z-update step
        zold = z.copy()
        ztilde = np.mean(beta_hat + np.array(u), axis=0)
        z = reg.proximal_operator(ztilde, lamduh / (rho * nchunks))
#        z = shrinkage(ztilde, lamduh / (rho * nchunks))

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


def local_update(X, y, beta, z, u, rho, f, fprime, solver=fmin_l_bfgs_b):

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


def bfgs(X, y, max_steps=500, tol=1e-14, family=Logistic):
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
    for k in range(max_steps):

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
        stepSize, _, _, func = compute_stepsize_dask(beta, step,
                                                     Xbeta, Xstep,
                                                     y, func, family=family,
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
            break

        df = lf - func
        df /= max(func, lf)
        if df < tol:
            print('Converged')
            break

    return beta


def proximal_grad(X, y, reg=L1, lamduh=0.1, family=Logistic,
                  max_steps=100, tol=1e-8, verbose=False):

    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)

    if verbose:
        print('#       -f        |df/f|    |dx/x|    step')
        print('----------------------------------------------')

    for k in range(max_steps):
        # Compute the gradient
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            func = family.loglike(Xbeta, y)

        gradient = family.gradient(Xbeta, X, y)

        Xbeta, func, gradient = persist(
            Xbeta, func, gradient)

        obeta = beta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = reg.proximal_operator(obeta - stepSize * gradient, stepSize * lamduh)
            step = obeta - beta
            Xbeta = X.dot(beta)

            overflow = (Xbeta < 700).all()
            overflow, Xbeta, beta = persist(overflow, Xbeta, beta)
            overflow = overflow.compute()

            # This prevents overflow
            if overflow:
                func = family.loglike(Xbeta, y)
                func = persist(func)[0]
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
        if verbose:
            print('%2d  %.6e %9.2e  %.2e  %.1e' % (k + 1, func, df, db, stepSize))
        if df < tol:
            print('Converged')
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta
