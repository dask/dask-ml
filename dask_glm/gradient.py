# Constants

import numpy as np
import dask.array as da


# Constants

firstBacktrackMult = 0.1
nextBacktrackMult = 0.5
armijoMult = 0.1
stepGrowth = 1.25
stepSize = 1.0
recalcRate = 10
backtrackMult = firstBacktrackMult


# Compute the initial point
def gradient(X, y, max_steps=100):
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
        steplen = (gradient ** 2).sum() ** 0.5
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
        if df < 1e-14:
            print('Converged')
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta

