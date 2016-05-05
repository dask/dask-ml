from __future__ import print_function, absolute_import, division

import dask.array as da
import numpy as np
from sklearn.linear_model.base import (BaseEstimator, SparseCoefMixin,
                                       LinearClassifierMixin)
from sklearn.utils.extmath import safe_sparse_dot, softmax

from .numpy_dask import dot, exp, log1p


class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
    """Logistic Regression classifier.

    Parameters
    ----------
    fit_intercept : bool, optional
        If True [default], a constant bias will be added to the decision
        function.
    max_iter : int, optional
        Maximum number of iterations taken for the solver to converge.
        Default is 100.
    verbose : bool, optional
        If True the solver will log status to stdout. Default is False.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-4.
    """
    def __init__(self, fit_intercept=True, max_iter=100, verbose=False,
                 tol=1e-4):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol

    def fit(self, X, y):
        w, c = gradient_descent(X, y, fit_intercept=self.fit_intercept,
                                max_iter=self.max_iter, tol=self.tol,
                                verbose=self.verbose)
        self.coef_ = w
        self.intercept_ = c
        return self

    def decision_function(self, X):
        return (safe_sparse_dot(X, self.coef_) + self.intercept_).ravel()

    def predict(self, X):
        return (self.decision_function(X) > 0).astype('i8')

    def predict_proba(self, X):
        return softmax(self.decision_function(X), copy=False)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))


def _intercept_dot(X, w, c):
    return dot(X, w) + c


def _intercept_norm(w, c):
    return ((w**2).sum() + c**2)**0.5


def gradient_descent(X, y, fit_intercept=True, max_iter=100, tol=1e-4,
                     verbose=False):
    """Solve a logistic regression problem using gradient descent"""

    log = print if verbose else lambda x: x
    # Tuning Params
    first_backtrack_mult = 0.1
    next_backtrack_mult = 0.5
    armijo_mult = 0.1
    step_growth = 1.25
    step_size = 1.0

    # Init
    w = np.zeros(X.shape[1])
    c = 0.0
    backtrack_mult = first_backtrack_mult

    log('##       -f        |df/f|    |dw/w|    step\n'
        '-------------------------------------------')
    for k in range(1, max_iter + 1):
        # Compute the gradient
        Xw = _intercept_dot(X, w, c)
        eXw = exp(Xw)
        f = log1p(eXw).sum() - dot(y, Xw)
        mult = eXw/(eXw + 1) - y
        grad = dot(X.T, mult)
        c_grad = mult.sum() if fit_intercept else 0.0
        Xgrad = _intercept_dot(X, grad, c_grad)

        Xw, f, grad, Xgrad, c_grad = da.compute(Xw, f, grad, Xgrad, c_grad)

        step_len = _intercept_norm(grad, c_grad)

        # Compute the step size using line search
        old_w = w
        old_c = c
        old_Xw = Xw
        old_f = f
        for ii in range(100):
            w = old_w - step_size * grad
            if fit_intercept:
                c = old_c - step_size * c_grad
            if ii and np.array_equal(w, old_w) and c == old_c:
                step_size = 0
                break
            Xw = old_Xw - step_size * Xgrad
            # This prevents overflow
            if np.all(Xw < 700):
                eXw = np.exp(Xw)
                f = np.log1p(eXw).sum() - np.dot(y, Xw)
                df = old_f - f
                if df >= armijo_mult * step_size * step_len**2:
                    break
            step_size *= backtrack_mult

        if step_size == 0:
            log('No more progress')
            break
        df /= max(f, old_f)
        dw = (step_size * step_len /
              (_intercept_norm(w, c) + step_size * step_len))
        log('%2d  %.6e %-.2e  %.2e  %.1e' % (k, f, df, dw, step_size))
        if df < tol:
            log('Converged')
            break
        step_size *= step_growth
        backtrack_mult = next_backtrack_mult
    else:
        log('Maximum Iterations')
    return w[:, None], c
