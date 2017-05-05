from __future__ import absolute_import, division, print_function

import numpy as np


class Regularizer(object):
    """Abstract base class for regularization object.

    Defines the set of methods required to create a new regularization object. This includes
    the regularization functions itself and its gradient, hessian, and proximal operator.
    """
    name = '_base'

    def f(self, beta):
        """Regularization function."""
        raise NotImplementedError

    def gradient(self, beta):
        """Gradient of regularization function."""
        raise NotImplementedError

    def hessian(self, beta):
        """Hessian of regularization function."""
        raise NotImplementedError

    def proximal_operator(self, beta, t):
        """Proximal operator for regularization function."""
        raise NotImplementedError

    def add_reg_f(self, f, lam):
        """Add regularization function to other function."""
        def wrapped(beta, *args):
            return f(beta, *args) + lam * self.f(beta)
        return wrapped

    def add_reg_grad(self, grad, lam):
        """Add regularization gradient to other gradient function."""
        def wrapped(beta, *args):
            return grad(beta, *args) + lam * self.gradient(beta)
        return wrapped

    def add_reg_hessian(self, hess, lam):
        """Add regularization hessian to other hessian function."""
        def wrapped(beta, *args):
            return hess(beta, *args) + lam * self.hessian(beta)
        return wrapped

    @classmethod
    def get(cls, obj):
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, str):
            return {o.name: o for o in cls.__subclasses__()}[obj]()
        raise TypeError('Not a valid regularizer object.')


class L2(Regularizer):
    """L2 regularization."""
    name = 'l2'

    def f(self, beta):
        return (beta**2).sum() / 2

    def gradient(self, beta):
        return beta

    def hessian(self, beta):
        return np.eye(len(beta))

    def proximal_operator(self, beta, t):
        return 1 / (1 + t) * beta


class L1(Regularizer):
    """L1 regularization."""
    name = 'l1'

    def f(self, beta):
        return (np.abs(beta)).sum()

    def gradient(self, beta):
        if np.any(np.isclose(beta, 0)):
            raise ValueError('l1 norm is not differentiable at 0!')
        else:
            return np.sign(beta)

    def hessian(self, beta):
        if np.any(np.isclose(beta, 0)):
            raise ValueError('l1 norm is not twice differentiable at 0!')
        return np.zeros((beta.shape[0], beta.shape[0]))

    def proximal_operator(self, beta, t):
        z = np.maximum(0, beta - t) - np.maximum(0, -beta - t)
        return z


class ElasticNet(Regularizer):
    """Elastic net regularization."""
    name = 'elastic_net'

    def __init__(self, weight=0.5):
        self.weight = weight
        self.l1 = L1()
        self.l2 = L2()

    def _weighted(self, left, right):
        return self.weight * left + (1 - self.weight) * right

    def f(self, beta):
        return self._weighted(self.l1.f(beta), self.l2.f(beta))

    def gradient(self, beta):
        return self._weighted(self.l1.gradient(beta), self.l2.gradient(beta))

    def hessian(self, beta):
        return self._weighted(self.l1.hessian(beta), self.l2.hessian(beta))

    def proximal_operator(self, beta, t):
        """See notebooks/ElasticNetProximalOperatorDerivation.ipynb for derivation."""
        g = self.weight * t

        @np.vectorize
        def func(b):
            if b <= g:
                return 0
            return (b - g * np.sign(b)) / (t - g + 1)
        return beta
