from __future__ import absolute_import, print_function, division

from collections import defaultdict

import dask
from dask.threaded import get as threaded_get
from dask.utils import derived_from

import numpy as np

from sklearn.base import (clone, is_classifier, BaseEstimator,
                          MetaEstimatorMixin)
from sklearn.exceptions import NotFittedError
from sklearn import model_selection
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import _check_param_grid, BaseSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.fixes import rankdata, MaskedArray
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import indexable, check_is_fitted

from .core import initialize_dask_graph, do_fit_and_score


class DaskBaseSearchCV(BaseEstimator, MetaEstimatorMixin):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True, get=None):
        self.scoring = scoring
        self.estimator = estimator
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.get = get

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['params'][self.best_index_]

    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['mean_test_score'][self.best_index_]

    def _check_is_fitted(self, method_name):
        if not self.refit:
            msg = ('This {0} instance was initialized with refit=False. {1} '
                   'is available only after refitting on the best '
                   'parameters.').format(type(self).__name__, method_name)
            raise NotFittedError(msg)
        else:
            check_is_fitted(self, 'best_estimator_')

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict_log_proba(self, X):
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def decision_function(self, X):
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def transform(self, X):
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def inverse_transform(self, Xt):
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.transform(Xt)

    @derived_from(BaseSearchCV)
    def score(self, X, y=None):
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        return self.scorer_(self.best_estimator_, X, y)

    def _fit(self, X, y=None, groups=None, **fit_params):
        if self.error_score != 'raise':
            raise NotImplementedError("`error_score` values other than "
                                      "`'raise'` are not implemented")

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())

        X, y, groups = indexable(X, y, groups)

        (dsk, X_train, y_train, X_test,
         y_test, n_splits) = initialize_dask_graph(estimator, X, y, cv, groups)

        keys = []
        for parameters in candidate_params:
            est = clone(estimator).set_params(**parameters)
            score = do_fit_and_score(dsk, est, X_train, y_train, X_test,
                                     y_test, n_splits, self.scorer_,
                                     fit_params=fit_params,
                                     return_train_score=self.return_train_score)
            keys.extend([[(s, n) for s in score] for n in range(n_splits)])

        # Store the graph and keys
        self.dask_graph_ = dsk
        self.dask_keys_ = keys

        # Compute the scores
        get = self.get or dask.context._globals.get('get') or threaded_get
        out = get(dsk, keys)

        if self.return_train_score:
            train_scores, test_scores, test_sample_counts = zip(*out)
        else:
            test_scores, test_sample_counts = zip(*out)

        # Construct the `cv_results_` dictionary
        results = {'params': candidate_params}
        n_candidates = len(candidate_params)
        test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=int)

        _store(results, 'test_score', test_scores, n_splits, n_candidates,
               splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store(results, 'train_score', train_scores,
                   n_splits, n_candidates, splits=True)

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(lambda: MaskedArray(np.empty(n_candidates),
                                                        mask=True,
                                                        dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]
        self.cv_results_ = results
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            best_parameters = candidate_params[self.best_index_]
            best = estimator.set_params(**best_parameters)
            best = (best.fit(X, y, **fit_params) if y is not None
                    else best.fit(X, **fit_params))
            self.best_estimator_ = best
        return self


def _store(results, key_name, array, n_splits, n_candidates,
           weights=None, splits=False, rank=False):
    """A small helper to store the scores/times to the cv_results_"""
    # When iterated first by parameters then by splits
    array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
    if splits:
        for split_i in range(n_splits):
            results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

    array_means = np.average(array, axis=1, weights=weights)
    results['mean_%s' % key_name] = array_means
    # Weighted std is not directly available in numpy
    array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2,
                                    axis=1, weights=weights))
    results['std_%s' % key_name] = array_stds

    if rank:
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-array_means, method='min'), dtype=np.int32)


class DaskGridSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, iid=True,
                 refit=True, cv=None, error_score='raise',
                 return_train_score=True, get=None):
        super(DaskGridSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get)

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return model_selection.ParameterGrid(self.param_grid)

    @derived_from(model_selection.GridSearchCV)
    def fit(self, X, y=None, groups=None, **fit_params):
        return self._fit(X, y=y, groups=groups, **fit_params)


class DaskRandomizedSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 iid=True, refit=True, cv=None, random_state=None,
                 error_score='raise', return_train_score=True, get=None):
        super(DaskRandomizedSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get)

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                self.n_iter, random_state=self.random_state)

    @derived_from(model_selection.RandomizedSearchCV)
    def fit(self, X, y=None, groups=None, **fit_params):
        return self._fit(X, y=y, groups=groups, **fit_params)
