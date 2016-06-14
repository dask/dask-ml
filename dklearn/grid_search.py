from __future__ import absolute_import, print_function, division

import numpy as np
from dask import threaded
from dask.base import tokenize
from dask.context import _globals
from dask.delayed import Delayed
from sklearn.base import is_classifier, BaseEstimator
from sklearn.cross_validation import check_cv
from sklearn.grid_search import (_CVScoreTuple, _check_param_grid,
                                 ParameterGrid, ParameterSampler)
from sklearn.metrics.scorer import check_scoring
from sklearn.utils import indexable, safe_indexing

from .core import from_sklearn, unpack_arguments


def _fit_and_score(estimator, X_name, y_name, scorer, train, test,
                   parameters, fit_params):
    estimator = estimator.set_params(**parameters)
    n_samples = len(test)

    # Extract train and test data
    token = tokenize(X_name, y_name, train, test)
    X_train_name = 'X_train-' + token
    y_train_name = 'y_train-' + token
    X_test_name = 'X_test-' + token
    y_test_name = 'y_test-' + token

    train, test, dsk = unpack_arguments(train, test)
    dsk[X_train_name] = (safe_indexing, X_name, train)
    dsk[y_train_name] = (safe_indexing, y_name, train)
    dsk[X_test_name] = (safe_indexing, X_name, test)
    dsk[y_test_name] = (safe_indexing, y_name, test)

    # Fit
    fit = estimator.fit(Delayed(X_train_name, [dsk]),
                        Delayed(y_train_name, [dsk]),
                        **fit_params)
    # Score
    score_name = 'score-' + tokenize(scorer, fit, X_test_name, y_test_name)
    fit_key, dsk2 = unpack_arguments(fit)
    dsk.update(dsk2)
    dsk[score_name] = (scorer, fit_key, X_test_name, y_test_name)

    return n_samples, score_name, dsk


def get_grid_scores(scores, parameters, n_samples, n_folds, iid):
    score_params_len = list(zip(scores, parameters, n_samples))
    n_fits = len(score_params_len)

    scores = list()
    grid_scores = list()
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


class BaseSearchCV(BaseEstimator):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, fit_params=None, iid=True,
                 refit=True, cv=None, get=None):
        self.scoring = scoring
        self.estimator = from_sklearn(estimator)
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.get = get

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def _fit(self, X, y, parameter_iterable):
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        n_folds = len(cv)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != len(X):
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), len(X)))

        # We tokenize X and y here to avoid repeating hashing
        X_name = 'X-' + tokenize(X)
        y_name = 'y-' + tokenize(y)
        dsk = {X_name: X, y_name: y}

        score_keys = []
        n_samples = []
        parameters = []
        for params in parameter_iterable:
            for train, test in cv:
                n, score_key, dsk2 = _fit_and_score(estimator, X_name, y_name,
                                                    self.scorer_, train, test,
                                                    params, self.fit_params)
                dsk.update(dsk2)
                score_keys.append(score_key)
                n_samples.append(n)
                parameters.append(params)

        # Compute results
        get = self.get or _globals['get'] or threaded.get
        scores = get(dsk, score_keys)

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
