from __future__ import print_function, absolute_import, division

import dask.array as da
import numpy as np

from .numpy_dask import dot, exp, log1p


def gradient(X, y, max_steps=100, verbose=True):
    """Solve a logistic regression problem using gradient descent"""

    log = print if verbose else lambda x: x
    M = X.shape[-1]
    first_backtrack_mult = 0.1
    next_backtrack_mult = 0.5
    armijo_mult = 0.1
    step_growth = 1.25
    step_size = 1.0
    recalc_rate = 10
    backtrack_mult = first_backtrack_mult
    beta = np.zeros(M)

    log('##       -f        |df/f|    |dx/x|    step')
    log('----------------------------------------------')
    for k in range(max_steps):
        # Compute the gradient
        if k % recalc_rate == 0:
            Xbeta = dot(X, beta)
            eXbeta = exp(Xbeta)
            func = log1p(eXbeta).sum() - dot(y, Xbeta)
        e1 = eXbeta + 1.0
        gradient = dot(X.T, eXbeta / e1 - y)
        steplen = (gradient**2).sum()**0.5
        Xgradient = dot(X, gradient)

        Xbeta, eXbeta, func, gradient, steplen, Xgradient =\
            da.compute(Xbeta, eXbeta, func, gradient, steplen, Xgradient)

        obeta = beta
        oXbeta = Xbeta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = obeta - step_size * gradient
            if ii and np.array_equal(beta, obeta):
                step_size = 0
                break
            Xbeta = oXbeta - step_size * Xgradient
            # This prevents overflow
            if np.all(Xbeta < 700):
                eXbeta = np.exp(Xbeta)
                func = np.sum(np.log1p(eXbeta)) - np.dot(y, Xbeta)
                df = lf - func
                if df >= armijo_mult * step_size * steplen ** 2:
                    break
            step_size *= backtrack_mult
        if step_size == 0:
            log('No more progress')
            break
        df /= max(func, lf)
        db = step_size * steplen / (np.linalg.norm(beta) + step_size * steplen)
        log('%2d  %.6e %-.2e  %.2e  %.1e' % (k + 1, func, df, db, step_size))
        if df < 1e-14:
            log('Converged')
            break
        step_size *= step_growth
        backtrack_mult = next_backtrack_mult

    return beta
