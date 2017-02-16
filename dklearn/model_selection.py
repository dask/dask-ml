from __future__ import absolute_import, print_function, division

from collections import defaultdict
from functools import partial

import numpy as np

import dask
from dask.threaded import get as threaded_get
from sklearn.base import (clone, is_classifier, BaseEstimator,
                          MetaEstimatorMixin)
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import (_check_param_grid, ParameterGrid,
                                             ParameterSampler)
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.fixes import rankdata, MaskedArray
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import indexable, check_is_fitted

from .core import initialize_dask_graph, do_fit_and_score


class BaseSearchCV(BaseEstimator, MetaEstimatorMixin):
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
    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.transform(Xt)

    def fit(self, X, y=None, groups=None, **fit_params):
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)

        X, y, groups = indexable(X, y, groups)

        (dsk, X_train, y_train,
         X_test, y_test, n_splits) = initialize_dask_graph(X, y, cv, groups)

        keys = []
        for parameters in candidate_params:
            est = clone(estimator).set_params(**parameters)
            score = do_fit_and_score(dsk, est, X_train, y_train, X_test,
                                     y_test, n_splits, self.scorer_,
                                     fit_params=fit_params,
                                     return_train_score=self.return_train_score,
                                     error_score=self.error_score)
            keys.extend([[(s, n) for s in score]
                         for n in range(n_splits)])

        # Store the graph and keys
        self.dask_graph_ = dsk
        self.dask_keys_ = keys

        # Compute the scores
        get = self.get or dask.context._globals.get('get') or threaded_get
        out = get(dsk, keys)

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            train_scores, test_scores, test_sample_counts = zip(*out)
        else:
            test_scores, test_sample_counts = zip(*out)

        results = dict()

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=int)

        _store(results, 'test_score', test_scores, n_splits, n_candidates,
               splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store(results, 'train_score', train_scores,
                   n_splits, n_candidates, splits=True)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
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


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, iid=True,
                 refit=True, cv=None, error_score='raise',
                 return_train_score=True, get=None):
        super(GridSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get)

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGrid(self.param_grid)


class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 iid=True, refit=True, cv=None, random_state=None,
                 error_score='raise', return_train_score=True, get=None):
        super(RandomizedSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get)

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=self.random_state)
