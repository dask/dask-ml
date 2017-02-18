from __future__ import absolute_import, division, print_function

from dask_glm.utils import dot


try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def _(func):
            return func
        return _


def loglike(Xbeta, y):
    return ((y - Xbeta)**2).sum()


def pointwise_loss(beta, X, y):
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    return loglike(Xbeta, y)


def pointwise_gradient(beta, X, y):
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    return gradient(Xbeta, X, y)


def gradient(Xbeta, X, y):
    return 2 *  dot(X.T, Xbeta) - 2 * dot(X.T, y)


def hessian(Xbeta, X):
    return dot(X.T, X)
