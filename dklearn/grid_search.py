from __future__ import absolute_import, print_function, division

import numpy as np
from dask import threaded
from dask.context import _globals
from dask.delayed import compute, delayed
from sklearn.base import is_classifier, BaseEstimator
from sklearn.grid_search import (_CVScoreTuple, _check_param_grid,
                                 ParameterGrid, ParameterSampler)
from sklearn.metrics.scorer import check_scoring

from .core import from_sklearn
from .cross_validation import check_cv
from .utils import check_X_y


def get_grid_scores(scores, parameters, n_samples, n_folds, iid):
    score_params_len = list(zip(scores, parameters, n_samples))
    n_fits = len(score_params_len)

    scores = []
    grid_scores = []
    for grid_start in range(0, n_fits, n_folds):
        n_test_samples = 0
        score = 0
        all_scores = []
        for this_score, parameters, this_n_test_samples in \
                score_params_len[grid_start:grid_start + n_folds]:
            all_scores.append(this_score)
            if iid:
                this_score *= this_n_test_samples
                n_test_samples += this_n_test_samples
            score += this_score
        if iid:
            score /= float(n_test_samples)
        else:
            score /= float(n_folds)
        scores.append((score, parameters))
        grid_scores.append(_CVScoreTuple(parameters, score,
                                         np.array(all_scores)))
    return grid_scores


def get_best(grid_scores):
    best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
    return best


@delayed(pure=True)
def score_and_n(scorer, fit, X_test, y_test):
    n = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
    score = scorer(fit, X_test, y_test)
    return score, n


class BaseSearchCV(BaseEstimator):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, fit_params=None, iid=True,
                 refit=True, cv=None, get=None):
        self.scoring = scoring
        self.estimator = estimator
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.get = get

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def _fit(self, X, y, parameter_iterable):
        estimator = from_sklearn(self.estimator)
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        cv = check_cv(self.cv, X, y, classifier=is_classifier(estimator))
        n_folds = len(cv)
        X, y = check_X_y(X, y)

        tups = []
        parameters = []
        train_test_sets = list(cv.split(X, y))
        for params in parameter_iterable:
            est = estimator.set_params(**params)
            for X_train, y_train, X_test, y_test in train_test_sets:
                fit = est.fit(X_train, y_train, **self.fit_params)
                tups.append(score_and_n(self.scorer_, fit, X_test, y_test))
                parameters.append(params)

        # Compute results
        get = self.get or _globals['get'] or threaded.get
        scores, n_samples = zip(*compute(tups, get=get)[0])

        # Extract grid_scores and best parameters
        grid_scores = get_grid_scores(scores, parameters, n_samples,
                                      n_folds, self.iid)
        best = get_best(grid_scores)

        # Update attributes
        self.grid_scores_ = grid_scores
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        # Refit if needed
        if self.refit:
            self.best_estimator_ = (estimator.set_params(**best.parameters)
                                             .fit(X, y, **self.fit_params)
                                             .compute(get=get))
        return self


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 iid=True, refit=True, cv=None, get=None):

        super(GridSearchCV, self).__init__(
                estimator=estimator, scoring=scoring, fit_params=fit_params,
                iid=iid, refit=refit, cv=cv, get=get)
        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def fit(self, X, y=None):
        return self._fit(X, y, ParameterGrid(self.param_grid))


class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, iid=True, refit=True, cv=None,
                 random_state=None, get=None):

        super(RandomizedSearchCV, self).__init__(
                estimator=estimator, scoring=scoring, fit_params=fit_params,
                iid=iid, refit=refit, cv=cv, get=get)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(X, y, sampled_params)
