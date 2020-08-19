# -*- coding: utf-8 -*-
"""Generalized Linear Models for large datasets."""
import textwrap

from dask_glm import algorithms, families
from dask_glm.utils import (
    accuracy_score,
    add_intercept,
    dot,
    exp,
    poisson_deviance,
    sigmoid,
)
from sklearn.base import BaseEstimator

from ..metrics import r2_score
from ..utils import check_array

_base_doc = textwrap.dedent(
    """\
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
        the parameterization :math:`\\lambda = 1 / C`

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

    max_iter : int, default 100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : str, default 'ovr'
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
    """
)


class _GLM(BaseEstimator):
    @property
    def family(self):
        """
        The family this estimator is for.
        """

    def __init__(
        self,
        penalty="l2",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight=None,
        random_state=None,
        solver="admm",
        max_iter=100,
        multi_class="ovr",
        verbose=0,
        warm_start=False,
        n_jobs=1,
        solver_kwargs=None,
    ):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.solver_kwargs = solver_kwargs

    def _get_solver_kwargs(self):
        fit_kwargs = {
            "max_iter": self.max_iter,
            "family": self.family,
            "tol": self.tol,
            "regularizer": self.penalty,
            "lamduh": 1 / self.C,
        }

        if self.solver in ("gradient_descent", "newton"):
            fit_kwargs.pop("regularizer")
            fit_kwargs.pop("lamduh")

        if self.solver == "admm":
            fit_kwargs.pop("tol")  # uses reltol / abstol instead

        if self.solver_kwargs:
            fit_kwargs.update(self.solver_kwargs)

        solvers = {
            "admm",
            "proximal_grad",
            "lbfgs",
            "newton",
            "proximal_grad",
            "gradient_descent",
        }

        if self.solver not in solvers:
            msg = "'solver' must be {}. Got '{}' instead".format(solvers, self.solver)
            raise ValueError(msg)

        return fit_kwargs

    def fit(self, X, y=None):
        """Fit the model on the training data

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self : objectj
        """
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

        return check_array(X, accept_unknown_chunks=True)


class LogisticRegression(_GLM):
    __doc__ = _base_doc.format(
        regression_type="logistic regression",
        examples=textwrap.dedent(
            """
            >>> from dask_glm.datasets import make_classification
            >>> X, y = make_classification()
            >>> lr = LogisticRegression()
            >>> lr.fit(X, y)
            >>> lr.decision_function(X)
            >>> lr.predict(X)
            >>> lr.predict_proba(X)
            >>> lr.score(X, y)"""
        ),
    )

    @property
    def family(self):
        return families.Logistic

    def decision_function(self, X):
        """Predict confidence scores for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            The confidence score of the sample for each class in the model.
        """
        X_ = self._check_array(X)
        return dot(X_, self._coef)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples,]
            Predicted class labels for each sample
        """
        return self.predict_proba(X) > 0.5  # TODO: verify, multi_class broken

    def predict_proba(self, X):
        """Probability estimates for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            The probability of the sample for each class in the model.
        """
        return sigmoid(self.decision_function(X))

    def score(self, X, y):
        """The mean accuracy on the given data and labels

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.
        y : array-like, shape = [n_samples,]
            Test labels.

        Returns
        -------
        score : float
            Mean accuracy score
        """
        return accuracy_score(y, self.predict(X))


class LinearRegression(_GLM):
    __doc__ = _base_doc.format(
        regression_type="linear regression",
        examples=textwrap.dedent(
            """
            >>> from dask_glm.datasets import make_regression
            >>> X, y = make_regression()
            >>> lr = LinearRegression()
            >>> lr.fit(X, y)
            >>> lr.predict(X)
            >>> lr.predict(X)
            >>> lr.score(X, y)"""
        ),
    )

    @property
    def family(self):
        return families.Normal

    def predict(self, X):
        """Predict values for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples,]
            Predicted value for each sample
        """
        X_ = self._check_array(X)
        return dot(X_, self._coef)

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return r2_score(y, self.predict(X))


class PoissonRegression(_GLM):
    __doc__ = _base_doc.format(
        regression_type="poisson regression",
        examples=textwrap.dedent(
            """
            >>> from dask_glm.datasets import make_counts
            >>> X, y = make_counts()
            >>> lr = PoissonRegression()
            >>> lr.fit(X, y)
            >>> lr.predict(X)
            >>> lr.predict(X)
            >>> lr.get_deviance(X, y)"""
        ),
    )

    @property
    def family(self):
        return families.Poisson

    def predict(self, X):
        """Predict count for samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples,]
            Predicted count for each sample
        """
        X_ = self._check_array(X)
        return exp(dot(X_, self._coef))

    def get_deviance(self, X, y):
        return poisson_deviance(y, self.predict(X))
