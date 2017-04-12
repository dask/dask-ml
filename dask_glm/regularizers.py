from __future__ import absolute_import, division, print_function

import numpy as np


class L2(object):

    @staticmethod
    def proximal_operator(beta, t):
        return 1 / (1 + t) * beta

    @staticmethod
    def hessian(beta):
        return 2 * np.eye(len(beta))

    @staticmethod
    def add_reg_hessian(hess, lam):
        def wrapped(beta, *args):
            return hess(beta, *args) + lam * L2.hessian(beta)
        return wrapped

    @staticmethod
    def f(beta):
        return (beta**2).sum()

    @staticmethod
    def add_reg_f(f, lam):
        def wrapped(beta, *args):
            return f(beta, *args) + lam * L2.f(beta)
        return wrapped

    @staticmethod
    def gradient(beta):
        return 2 * beta

    @staticmethod
    def add_reg_grad(grad, lam):
        def wrapped(beta, *args):
            return grad(beta, *args) + lam * L2.gradient(beta)
        return wrapped


class L1(object):

    @staticmethod
    def proximal_operator(beta, t):
        z = np.maximum(0, beta - t) - np.maximum(0, -beta - t)
        return z

    @staticmethod
    def hessian(beta):
        raise ValueError('l1 norm is not twice differentiable!')

    @staticmethod
    def add_reg_hessian(hess, lam):
        def wrapped(beta, *args):
            return hess(beta, *args) + lam * L1.hessian(beta)
        return wrapped

    @staticmethod
    def f(beta):
        return (np.abs(beta)).sum()

    @staticmethod
    def add_reg_f(f, lam):
        def wrapped(beta, *args):
            return f(beta, *args) + lam * L1.f(beta)
        return wrapped

    @staticmethod
    def gradient(beta):
        if np.any(np.isclose(beta, 0)):
            raise ValueError('l1 norm is not differentiable at 0!')
        else:
            return np.sign(beta)

    @staticmethod
    def add_reg_grad(grad, lam):
        def wrapped(beta, *args):
            return grad(beta, *args) + lam * L1.gradient(beta)
        return wrapped


_regularizers = {
    'l1': L1,
    'l2': L2,
}
