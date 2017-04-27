"""
Models following scikit-learn's estimator API.
"""
from sklearn.base import BaseEstimator

from . import algorithms
from . import families
from .utils import (
    sigmoid, dot, add_intercept, mean_squared_error, accuracy_score
)


class _GLM(BaseEstimator):
    @property
    def family(self):
        """
        Family
        """

    def __init__(self, fit_intercept=True, solver='admm', regularizer='l2',
                 max_iter=100, tol=1e-4, lamduh=1.0, rho=1,
                 over_relax=1, abstol=1e-4, reltol=1e-2):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.regularizer = regularizer
        self.max_iter = max_iter
        self.tol = tol
        self.lamduh = lamduh
        self.rho = rho
        self.over_relax = over_relax
        self.abstol = abstol
        self.reltol = reltol

        self.coef_ = None
        self.intercept_ = None
        self._coef = None  # coef, maybe with intercept

        fit_kwargs = {'max_iter', 'tol', 'family'}

        if solver == 'admm':
            fit_kwargs.discard('tol')
            fit_kwargs.update({
                'regularizer', 'lamduh', 'rho', 'over_relax', 'abstol',
                'reltol'
            })
        elif solver == 'proximal_grad':
            fit_kwargs.update({'regularizer', 'lamduh'})

        self._fit_kwargs = {k: getattr(self, k) for k in fit_kwargs}

    def fit(self, X, y=None):
        X_ = self._maybe_add_intercept(X)
        self._coef = algorithms._solvers[self.solver](X_, y, **self._fit_kwargs)

        if self.fit_intercept:
            self.coef_ = self._coef[:-1]
            self.intercept_ = self._coef[-1]
        else:
            self.coef_ = self._coef
        return self

    def _maybe_add_intercept(self, X):
        if self.fit_intercept:
            return add_intercept(X)
        else:
            return X


class LogisticRegression(_GLM):
    """
    Parameters
    ----------
    fit_intercept : bool, default True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    solver : {'admm', 'gradient_descent', 'newton', 'bfgs', 'proximal_grad'}
        Solver to use. See :ref:`algorithms` for details
    regularizer : {'l1', 'l2'}
        Regularizer to use. See :ref:`regularizers` for details.
        Only used with ``admm`` and ``proximal_grad`` solvers.
    max_iter : int, default 100
        Maximum number of iterations taken for the solvers to converge
    tol : float, default 1e-4
        Tolerance for stopping criteria. Ignored for ``admm`` solver
    lambduh : float, default 1.0
        Only used with ``admm`` and ``proximal_grad`` solvers
    rho, over_relax, abstol, reltol : float
        Only used with the ``admm`` solver. See :ref:`algorithms.admm`
        for details

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
    intercept_ : float

    Examples
    --------
    >>> from dask_glm.datasets import make_classification
    >>> X, y = make_classification()
    >>> lr = LogisticRegression()
    >>> lr.fit(X, y)
    >>> lr.predict(X)
    >>> lr.predict_proba(X)
    """

    @property
    def family(self):
        return families.Logistic

    def predict(self, X):
        return self.predict_proba(X) > .5  # TODO: verify, multiclass broken

    def predict_proba(self, X):
        X_ = self._maybe_add_intercept(X)
        return sigmoid(dot(X_, self._coef))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class LinearRegression(_GLM):
    """
    Ordinary Lest Square regression
    """
    @property
    def family(self):
        return families.Normal

    def predict(self, X):
        X_ = self._maybe_add_intercept(X)
        return dot(X_, self._coef)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))
