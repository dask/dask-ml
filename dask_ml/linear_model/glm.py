"""Generalized Linear Models for large datasets."""
import textwrap

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


_base_doc = textwrap.dedent("""\
    Esimator for {regression_type}.

    Parameters
    ----------
    penalty : str or Regularizer, default 'l2'
        Regularizer to use. Only relevant for the 'admm', 'lbfgs' and
        'proximal_grad' solvers.

        For string values, only 'l1' or 'l2' are valid.

    dual : bool
        Ignored

    tol : float, default 1e-4
        The tolerance for convergence.

    C : float
        Regularization strength. Note that ``dask-glm`` solvers use
        the parameterization :math:`lamduh = 1 / C`

    fit_intercept : bool, default True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : bool
        Ignored

    class_weight : dict or 'balanced'
        Ignored

    random_state : int, RandomState, or None

        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.

    solver : {{'admm', 'gradient_descent', 'newton', 'lbfgs', 'proximal_grad'}}
        Solver to use. See :ref:`api.algorithms` for details

    multiclass : str, default 'ovr'
        Ignored. Multiclass solvers not currently supported.

    verbose : int, default 0
        Ignored

    warm_start : bool, default False
        Ignored

    n_jobs : int, default 1
        Ignored

    solver_kwargs : dict, optional, default None
        Extra keyword arguments to pass through to the solver.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The learned value for the model's coefficients
    intercept_ : float of None
        The learned value for the intercept, if one was added
        to the model

    Examples
    --------
    {examples}
    """)


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
                 max_iter=100,
                 solver_kwargs=None):
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
        self.solver_kwargs = solver_kwargs

    def _get_solver_kwargs(self):
        fit_kwargs = {'max_iter': self.max_iter,
                      'family': self.family,
                      'tol': self.tol,
                      'regularizer': self.penalty,
                      'lamduh': 1 / self.C}

        if self.solver in ('gradient_descent', 'newton'):
            fit_kwargs.pop('regularizer')
            fit_kwargs.pop('lamduh')

        if self.solver == 'admm':
            fit_kwargs.pop('tol')  # uses reltol / abstol instead

        if self.solver_kwargs:
            fit_kwargs.update(self.solver_kwargs)

        solvers = {'admm', 'proximal_grad', 'lbfgs', 'newton',
                   'proximal_grad', 'gradient_descent'}

        if self.solver not in solvers:
            msg = ("'solver' must be {}. Got '{}' instead".format(solvers,
                                                                  self.solver))
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
    __doc__ = _base_doc.format(
        regression_type='logistic_regression',
        examples=textwrap.dedent("""
            >>> from dask_glm.datasets import make_classification
            >>> X, y = make_classification()
            >>> lr = LogisticRegression()
            >>> lr.fit(X, y)
            >>> lr.predict(X)
            >>> lr.predict_proba(X)
            >>> est.score(X, y)"""))

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
    __doc__ = _base_doc.format(
        regression_type='linear_regression',
        examples=textwrap.dedent("""
            >>> from dask_glm.datasets import make_regression
            >>> X, y = make_regression()
            >>> lr = LinearRegression()
            >>> lr.fit(X, y)
            >>> lr.predict(X)
            >>> lr.predict(X)
            >>> est.score(X, y)"""))

    @property
    def family(self):
        return families.Normal

    def predict(self, X):
        X_ = self._check_array(X)
        return dot(X_, self._coef)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))


class PoissonRegression(_GLM):
    __doc__ = _base_doc.format(
        regression_type='poisson_regression',
        examples=textwrap.dedent("""
            >>> from dask_glm.datasets import make_counts
            >>> X, y = make_counts()
            >>> lr = PoissonRegression()
            >>> lr.fit(X, y)
            >>> lr.predict(X)
            >>> lr.predict(X)
            >>> lr.get_deviance(X, y)"""))

    @property
    def family(self):
        return families.Poisson

    def predict(self, X):
        X_ = self._check_array(X)
        return exp(dot(X_, self._coef))

    def get_deviance(self, X, y):
        return poisson_deviance(y, self.predict(X))
