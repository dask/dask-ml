"""Generalized Linear Models for large datasets."""
from sklearn.base import BaseEstimator

from dask_glm import algorithms
from dask_glm import families
from dask_glm.utils import (
    sigmoid, dot, add_intercept, mean_squared_error, accuracy_score, exp,
    poisson_deviance
)

# register multipledispatch
from . import utils  # noqa
from ..utils import check_array


class _GLM(BaseEstimator):

    @property
    def family(self):
        """
        The family this estimator is for.
        """

    def __init__(self,
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 class_weight=None,
                 random_state=None,
                 solver='admm',
                 multiclass='ovr',
                 verbose=0,
                 warm_start=False,
                 n_jobs=1,
                 max_iter=100):
        """

        """
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.multiclass = multiclass
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.solver_kwargs = None

    def _get_solver_kwargs(self):
        fit_kwargs = {'max_iter': self.max_iter,
                      'family': self.family,
                      'tol': self.tol,
                      'regularizer': self.penalty,
                      'lamduh': 1 / self.C}
        if self.solver_kwargs:
            fit_kwargs.update(self.solver_kwargs)
        if self.solver not in {'admm', 'proximal_grad', 'lbfgs'}:
            msg = ("'solver' must be one of 'admm', 'proximal_grad' or "
                   "'lbfgs'. Got {} instead.".format(self.solver))
            raise ValueError(msg)

        return fit_kwargs

    def fit(self, X, y=None):
        X = self._check_array(X)

        solver_kwargs = self._get_solver_kwargs()

        self._coef = algorithms._solvers[self.solver](X, y, **solver_kwargs)
        if self.fit_intercept:
            self.coef_ = self._coef[:-1]
            self.intercept_ = self._coef[-1]
        else:
            self.coef_ = self._coef
        return self

    def _check_array(self, X):
        if self.fit_intercept:
            X = add_intercept(X)

        return check_array(X)


class LogisticRegression(_GLM):
    """
    Esimator for logistic regression.

    Parameters
    ----------
    fit_intercept : bool, default True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    solver : {'admm', 'gradient_descent', 'newton', 'lbfgs', 'proximal_grad'}
        Solver to use. See :ref:`api.algorithms` for details
    regularizer : {'l1', 'l2'}
        Regularizer to use. See :ref:`api.regularizers` for details.
        Only used with ``admm``, ``lbfgs``, and ``proximal_grad`` solvers.
    max_iter : int, default 100
        Maximum number of iterations taken for the solvers to converge
    tol : float, default 1e-4
        Tolerance for stopping criteria. Ignored for ``admm`` solver
    C : float, default 1.0
        Only used with ``admm``, ``lbfgs`` and ``proximal_grad`` solvers.
    rho, over_relax, abstol, reltol : float
        Only used with the ``admm`` solver.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The learned value for the model's coefficients
    intercept_ : float of None
        The learned value for the intercept, if one was added
        to the model

    Examples
    --------
    >>> from dask_glm.datasets import make_classification
    >>> X, y = make_classification()
    >>> lr = LogisticRegression()
    >>> lr.fit(X, y)
    >>> lr.predict(X)
    >>> lr.predict_proba(X)
    >>> est.score(X, y)
    """

    @property
    def family(self):
        return families.Logistic

    def predict(self, X):
        return self.predict_proba(X) > .5  # TODO: verify, multiclass broken

    def predict_proba(self, X):
        X_ = self._check_array(X)
        return sigmoid(dot(X_, self._coef))

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class LinearRegression(_GLM):
    """
    Esimator for a linear model using Ordinary Least Squares.

    Parameters
    ----------
    fit_intercept : bool, default True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    solver : {'admm', 'gradient_descent', 'newton', 'lbfgs', 'proximal_grad'}
        Solver to use. See :ref:`api.algorithms` for details
    regularizer : {'l1', 'l2'}
        Regularizer to use. See :ref:`api.regularizers` for details.
        Only used with ``admm`` and ``proximal_grad`` solvers.
    max_iter : int, default 100
        Maximum number of iterations taken for the solvers to converge
    tol : float, default 1e-4
        Tolerance for stopping criteria. Ignored for ``admm`` solver
    C : float, default 1.0
        Only used with ``admm`` and ``proximal_grad`` solvers
    rho, over_relax, abstol, reltol : float
        Only used with the ``admm`` solver.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The learned value for the model's coefficients
    intercept_ : float of None
        The learned value for the intercept, if one was added
        to the model

    Examples
    --------
    >>> from dask_glm.datasets import make_regression
    >>> X, y = make_regression()
    >>> est = LinearRegression()
    >>> est.fit(X, y)
    >>> est.predict(X)
    >>> est.score(X, y)
    """
    @property
    def family(self):
        return families.Normal

    def predict(self, X):
        X_ = self._check_array(X)
        return dot(X_, self._coef)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


class PoissonRegression(_GLM):
    """
    Esimator for Poisson Regression.

    Parameters
    ----------
    fit_intercept : bool, default True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    solver : {'admm', 'gradient_descent', 'newton', 'lbfgs', 'proximal_grad'}
        Solver to use. See :ref:`api.algorithms` for details
    regularizer : {'l1', 'l2'}
        Regularizer to use. See :ref:`api.regularizers` for details.
        Only used with ``admm``, ``lbfgs``, and ``proximal_grad`` solvers.
    max_iter : int, default 100
        Maximum number of iterations taken for the solvers to converge
    tol : float, default 1e-4
        Tolerance for stopping criteria. Ignored for ``admm`` solver
    C : float, default 1.0
        Only used with ``admm``, ``lbfgs`` and ``proximal_grad`` solvers.
    rho, over_relax, abstol, reltol : float
        Only used with the ``admm`` solver.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The learned value for the model's coefficients
    intercept_ : float of None
        The learned value for the intercept, if one was added
        to the model

    Examples
    --------
    >>> from dask_glm.datasets import make_poisson
    >>> X, y = make_poisson()
    >>> pr = PoissonRegression()
    >>> pr.fit(X, y)
    >>> pr.predict(X)
    >>> pr.get_deviance(X, y)
    """
    @property
    def family(self):
        return families.Poisson

    def predict(self, X):
        X_ = self._check_array(X)
        return exp(dot(X_, self._coef))

    def get_deviance(self, X, y):
        return poisson_deviance(y, self.predict(X))
