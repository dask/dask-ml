from __future__ import absolute_import, division, print_function

from dask_glm.utils import dot, exp, log1p, sigmoid


try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def _(func):
            return func
        return _


def loglike(Xbeta, y):
    eXbeta = exp(Xbeta)
    return (log1p(eXbeta)).sum() - dot(y, Xbeta)


def pointwise_loss(beta, X, y):
    '''Logistic Loss, evaluated point-wise.'''
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    return loglike(Xbeta, y)


def pointwise_gradient(beta, X, y):
    '''Logistic gradient, evaluated point-wise.'''
    beta, y = beta.ravel(), y.ravel()
    Xbeta = X.dot(beta)
    return gradient(Xbeta, X, y)


def gradient(Xbeta, X, y):
    p = sigmoid(Xbeta)
    return dot(X.T, p-y)


def hessian(Xbeta, X):
    p = sigmoid(Xbeta)
    return dot(p * (1- p ) * X.T, X)
