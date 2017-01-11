from __future__ import absolute_import, division, print_function

from dask_glm.utils import *
import dask.array as da
import dask.dataframe as dd
import numpy as np

def bfgs(X, y, max_iter=50, tol=1e-14):
    '''Simple implementation of BFGS.'''

    n, p = X.shape

    recalcRate = 10
    stepSize = 1.0
    armijoMult = 1e-4
    backtrackMult = 0.1

    beta = np.zeros(p)
    Hk = np.eye(p)

    for k in range(max_iter):

        if k % recalcRate==0:
            Xbeta = X.dot(beta)
            eXbeta = exp(Xbeta)
            func = log1p(eXbeta).sum() - y.dot(Xbeta)

        e1 = eXbeta + 1.0
        gradient = X.T.dot(eXbeta / e1 - y)

        if k:
            yk += gradient
            rhok = 1/yk.dot(sk)
            adj = np.eye(p) - rhok*sk.dot(yk.T)
            Hk = adj.dot(Hk.dot(adj.T)) + rhok*sk.dot(sk.T)

        step = Hk.dot(gradient)
        steplen = step.dot(gradient)
        Xstep = X.dot(step)

        Xbeta, gradient, func, steplen, step, Xstep = da.compute(
                Xbeta, gradient, func, steplen, step, Xstep)

        # Compute the step size
        lf = func
        obeta = beta
        oXbeta = Xbeta

        for ii in range(100):
            beta = obeta - stepSize * step
            if ii and np.array_equal(beta, obeta):
                stepSize = 0
                break
            Xbeta = oXbeta - stepSize * Xstep

            # This prevents overflow
            if np.all(Xbeta < 700):
                eXbeta = np.exp(Xbeta)
                func = np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)
                df = lf - func
                if df >= armijoMult * stepSize * steplen:
                    break
            stepSize *= backtrackMult

        yk = -gradient
        sk = -stepSize*step
        stepSize = 1.0

        if stepSize == 0:
            if verbose:
                print('No more progress')

        df /= max(func, lf)
        if df < tol:
            print('Converged')
            break

    return beta

def gradient_descent(X, y, max_steps=100, tol=1e-14):
    '''Michael Grant's implementation of Gradient Descent.'''

    N, M = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(M)

    print('##       -f        |df/f|    |dx/x|    step')
    print('----------------------------------------------')
    for k in range(max_steps):
        # Compute the gradient
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            eXbeta = da.exp(Xbeta)
            func = da.log1p(eXbeta).sum() - y.dot(Xbeta)
        e1 = eXbeta + 1.0
        gradient = X.T.dot(eXbeta / e1 - y)
        steplen = (gradient**2).sum()**0.5
        Xgradient = X.dot(gradient)

        Xbeta, eXbeta, func, gradient, steplen, Xgradient = da.compute(
                Xbeta, eXbeta, func, gradient, steplen, Xgradient)

        obeta = beta
        oXbeta = Xbeta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = obeta - stepSize * gradient
            if ii and np.array_equal(beta, obeta):
                stepSize = 0
                break
            Xbeta = oXbeta - stepSize * Xgradient
            # This prevents overflow
            if np.all(Xbeta < 700):
                eXbeta = np.exp(Xbeta)
                func = np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)
                df = lf - func
                if df >= armijoMult * stepSize * steplen ** 2:
                    break
            stepSize *= backtrackMult
        if stepSize == 0:
            print('No more progress')
            break
        df /= max(func, lf)
        db = stepSize * steplen / (np.linalg.norm(beta) + stepSize * steplen)
        print('%2d  %.6e %9.2e  %.2e  %.1e' % (k + 1, func, df, db, stepSize))
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

        ## should this use map_blocks()?
        p = sigmoid(Xbeta)
        hessian = dot(p*(1-p)*X.T, X)
        grad = X.T.dot(p-y)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, *_ = da.linalg.lstsq(hessian, grad)
        beta = (beta_old - step).compute()

        Xbeta = X.dot(beta)
        iter_count += 1
        
        ## should change this criterion
        coef_change = np.absolute(beta_old - beta)
        converged = ((not np.any(coef_change>tol)) or (iter_count>max_iter))

    return beta

def proximal_grad(X, y, reg='l2', max_iter=50, tol=1e-8):
    raise NotImplementedError
