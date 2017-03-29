from __future__ import absolute_import, division, print_function

from operator import getitem
from collections import defaultdict
from itertools import repeat
import numbers

import numpy as np
import dask
from dask.base import tokenize, Base
from dask.delayed import delayed
from dask.threaded import get as threaded_get
from dask.utils import derived_from
from sklearn import model_selection
from sklearn.base import is_classifier, clone, BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._search import _check_param_grid, BaseSearchCV
from sklearn.model_selection._split import (_BaseKFold,
                                            BaseShuffleSplit,
                                            KFold,
                                            StratifiedKFold,
                                            LeaveOneOut,
                                            LeaveOneGroupOut,
                                            LeavePOut,
                                            LeavePGroupsOut,
                                            PredefinedSplit,
                                            _CVIterableWrapper)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_is_fitted

from ._normalize import normalize_estimator
from .methods import (fit, fit_transform, fit_and_score, pipeline, fit_best,
                      get_best_params, create_cv_results, cv_split,
                      cv_n_samples, cv_extract, cv_extract_params,
                      decompress_params, score, feature_union,
                      feature_union_concat, MISSING)
from .utils import to_indexable, to_keys, unzip

try:
    from cytoolz import get, pluck
except:  # pragma: no cover
    from toolz import get, pluck


__all__ = ['GridSearchCV', 'RandomizedSearchCV']


class TokenIterator(object):
    def __init__(self, base_token):
        self.token = base_token
        self.counts = defaultdict(int)

    def __call__(self, est):
        typ = type(est)
        c = self.counts[typ]
        self.counts[typ] += 1
        return self.token if c == 0 else self.token + str(c)


def build_graph(estimator, cv, scorer, candidate_params, X, y=None,
                groups=None, fit_params=None, iid=True, refit=True,
                error_score='raise', return_train_score=True, cache_cv=True):

    X, y, groups = to_indexable(X, y, groups)
    cv = check_cv(cv, y, is_classifier(estimator))
    # "pairwise" estimators require a different graph for CV splitting
    is_pairwise = getattr(estimator, '_pairwise', False)

    dsk = {}
    X_name, y_name, groups_name = to_keys(dsk, X, y, groups)
    n_splits = compute_n_splits(cv, X, y, groups)

    if fit_params:
        # A mapping of {name: (name, graph-key)}
        param_values = to_indexable(*fit_params.values(), allow_scalars=True)
        fit_params = {k: (k, v) for (k, v) in
                      zip(fit_params, to_keys(dsk, *param_values))}
    else:
        fit_params = {}

    fields, tokens, params = normalize_params(candidate_params)
    main_token = tokenize(normalize_estimator(estimator), fields, params,
                          X_name, y_name, groups_name, fit_params, cv,
                          error_score == 'raise', return_train_score)

    cv_name = 'cv-split-' + main_token
    dsk[cv_name] = (cv_split, cv, X_name, y_name, groups_name,
                    is_pairwise, cache_cv)

    if iid:
        weights = 'cv-n-samples-' + main_token
        dsk[weights] = (cv_n_samples, cv_name)
    else:
        weights = None

    scores = do_fit_and_score(dsk, main_token, estimator, cv_name, fields,
                              tokens, params, X_name, y_name, fit_params,
                              n_splits, error_score, scorer,
                              return_train_score)

    cv_results = 'cv-results-' + main_token
    candidate_params_name = 'cv-parameters-' + main_token
    dsk[candidate_params_name] = (decompress_params, fields, params)
    dsk[cv_results] = (create_cv_results, scores, candidate_params_name,
                       n_splits, error_score, weights)
    keys = [cv_results]

    if refit:
        best_params = 'best-params-' + main_token
        dsk[best_params] = (get_best_params, candidate_params_name, cv_results)
        best_estimator = 'best-estimator-' + main_token
        if fit_params:
            fit_params = (dict, (zip, list(fit_params.keys()),
                                list(pluck(1, fit_params.values()))))
        dsk[best_estimator] = (fit_best, clone(estimator), best_params,
                               X_name, y_name, fit_params)
        keys.append(best_estimator)

    return dsk, keys, n_splits


def normalize_params(params):
    """Take a list of dictionaries, and tokenize/normalize."""
    # Collect a set of all fields
    fields = set()
    for p in params:
        fields.update(p)
    fields = sorted(fields)

    params2 = list(pluck(fields, params, MISSING))
    # Non-basic types (including MISSING) are unique to their id
    tokens = [tuple(x if isinstance(x, (int, float, str)) else id(x)
                    for x in p) for p in params2]

    return fields, tokens, params2


def _get_fit_params(cv, fit_params, n_splits):
    if not fit_params:
        return [(n, None) for n in range(n_splits)]
    keys = []
    vals = []
    for name, (full_name, val) in fit_params.items():
        vals.append(val)
        keys.append((name, full_name))
    return [(n, (cv_extract_params, cv, keys, vals, n))
            for n in range(n_splits)]


def _group_fit_params(steps, fit_params):
    param_lk = {n: {} for n, _ in steps}
    for pname, pval in fit_params.items():
        step, param = pname.split('__', 1)
        param_lk[step][param] = pval
    return param_lk


def do_fit_and_score(dsk, main_token, est, cv, fields, tokens, params,
                     X, y, fit_params, n_splits, error_score, scorer,
                     return_train_score):
    if not isinstance(est, Pipeline):
        # Fitting and scoring can all be done as a single task
        n_and_fit_params = _get_fit_params(cv, fit_params, n_splits)

        est_type = type(est).__name__.lower()
        est_name = '%s-%s' % (est_type, main_token)
        score_name = '%s-fit-score-%s' % (est_type, main_token)
        dsk[est_name] = est

        seen = {}
        m = 0
        out = []
        out_append = out.append

        for t, p in zip(tokens, params):
            if t in seen:
                out_append(seen[t])
            else:
                for n, fit_params in n_and_fit_params:
                    dsk[(score_name, m, n)] = (fit_and_score, est_name, cv,
                                               X, y, n, scorer, error_score,
                                               fields, p, fit_params,
                                               return_train_score)
                seen[t] = (score_name, m)
                out_append((score_name, m))
                m += 1
        scores = [k + (n,) for n in range(n_splits) for k in out]
    else:
        X_train = (cv_extract, cv, X, y, True, True)
        X_test = (cv_extract, cv, X, y, True, False)
        y_train = (cv_extract, cv, X, y, False, True)
        y_test = (cv_extract, cv, X, y, False, False)

        # Fit the estimator on the training data
        X_trains = [X_train] * len(params)
        y_trains = [y_train] * len(params)
        fit_ests = do_fit(dsk, TokenIterator(main_token), est, cv,
                          fields, tokens, params, X_trains, y_trains,
                          fit_params, n_splits, error_score)

        score_name = 'score-' + main_token

        scores = []
        scores_append = scores.append
        for n in range(n_splits):
            if return_train_score:
                xtrain = X_train + (n,)
                ytrain = y_train + (n,)
            else:
                xtrain = ytrain = None

            xtest = X_test + (n,)
            ytest = y_test + (n,)

            for (name, m) in fit_ests:
                dsk[(score_name, m, n)] = (score, (name, m, n),
                                        xtest, ytest, xtrain, ytrain, scorer)
                scores_append((score_name, m, n))
    return scores


def do_fit(dsk, next_token, est, cv, fields, tokens, params, Xs, ys,
           fit_params, n_splits, error_score):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, next_token, est, cv, fields, tokens, params,
                            Xs, ys, fit_params, n_splits, error_score, False)
    else:
        n_and_fit_params = _get_fit_params(cv, fit_params, n_splits)

        if params is None:
            params = tokens = repeat(None)
            fields = None

        token = next_token(est)
        est_type = type(est).__name__.lower()
        est_name = '%s-%s' % (est_type, token)
        fit_name = '%s-fit-%s' % (est_type, token)
        dsk[est_name] = est

        seen = {}
        m = 0
        out = []
        out_append = out.append

        for X, y, t, p in zip(Xs, ys, tokens, params):
            if (X, y, t) in seen:
                out_append(seen[X, y, t])
            else:
                for n, fit_params in n_and_fit_params:
                    dsk[(fit_name, m, n)] = (fit, est_name, X + (n,),
                                             y + (n,), error_score,
                                             fields, p, fit_params)
                seen[(X, y, t)] = (fit_name, m)
                out_append((fit_name, m))
                m += 1

        return out


def do_fit_transform(dsk, next_token, est, cv, fields, tokens, params, Xs, ys,
                     fit_params, n_splits, error_score):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, next_token, est, cv, fields, tokens, params,
                            Xs, ys, fit_params, n_splits, error_score, True)
    elif isinstance(est, FeatureUnion) and params is not None:
        return _do_featureunion(dsk, next_token, est, cv, fields, tokens,
                                params, Xs, ys, fit_params, n_splits,
                                error_score)
    else:
        n_and_fit_params = _get_fit_params(cv, fit_params, n_splits)

        if params is None:
            params = tokens = repeat(None)
            fields = None

        name = type(est).__name__.lower()
        token = next_token(est)
        fit_Xt_name = '%s-fit-transform-%s' % (name, token)
        fit_name = '%s-fit-%s' % (name, token)
        Xt_name = '%s-transform-%s' % (name, token)
        est_name = '%s-%s' % (type(est).__name__.lower(), token)
        dsk[est_name] = est

        seen = {}
        m = 0
        out = []
        out_append = out.append

        for X, y, t, p in zip(Xs, ys, tokens, params):
            if (X, y, t) in seen:
                out_append(seen[X, y, t])
            else:
                for n, fit_params in n_and_fit_params:
                    dsk[(fit_Xt_name, m, n)] = (fit_transform, est_name,
                                                X + (n,), y + (n,),
                                                error_score, fields, p,
                                                fit_params)
                    dsk[(fit_name, m, n)] = (getitem, (fit_Xt_name, m, n), 0)
                    dsk[(Xt_name, m, n)] = (getitem, (fit_Xt_name, m, n), 1)
                seen[X, y, t] = m
                out_append(m)
                m += 1

        return [(fit_name, i) for i in out], [(Xt_name, i) for i in out]


def _group_subparams(steps, fields, ignore=()):
    # Group the fields into a mapping of {stepname: [(newname, orig_index)]}
    field_to_index = dict(zip(fields, range(len(fields))))
    step_fields_lk = {s: [] for s, _ in steps}
    for f in fields:
        if '__' in f:
            step, param = f.split('__', 1)
            if step in step_fields_lk:
                step_fields_lk[step].append((param, field_to_index[f]))
                continue
        if f not in step_fields_lk and f not in ignore:
            raise ValueError("Unknown parameter: `%s`" % f)
    return field_to_index, step_fields_lk


def _group_ids_by_index(index, tokens):
    id_groups = []

    def new_group():
        o = []
        id_groups.append(o)
        return o.append

    _id_groups = defaultdict(new_group)
    for n, t in enumerate(pluck(index, tokens)):
        _id_groups[t](n)
    return id_groups


def _do_fit_step(dsk, next_token, step, cv, fields, tokens, params, Xs, ys,
                 fit_params, n_splits, error_score, step_fields_lk,
                 fit_params_lk, field_to_index, step_name, none_passthrough,
                 is_transform):
    sub_fields, sub_inds = map(list, unzip(step_fields_lk[step_name], 2))
    sub_fit_params = fit_params_lk[step_name]

    if step_name in field_to_index:
        # The estimator may change each call
        new_fits = {}
        new_Xs = {}
        est_index = field_to_index[step_name]

        for ids in _group_ids_by_index(est_index, tokens):
            # Get the estimator for this subgroup
            sub_est = params[ids[0]][est_index]
            if sub_est is MISSING:
                sub_est = step

            # If an estimator is `None`, there's nothing to do
            if sub_est is None:
                nones = dict.fromkeys(ids, None)
                new_fits.update(nones)
                if is_transform:
                    if none_passthrough:
                        new_Xs.update(zip(ids, get(ids, Xs)))
                    else:
                        new_Xs.update(nones)
            else:
                # Extract the proper subset of Xs, ys
                sub_Xs = get(ids, Xs)
                sub_ys = get(ids, ys)
                # Only subset the parameters/tokens if necessary
                if sub_fields:
                    sub_tokens = list(pluck(sub_inds, get(ids, tokens)))
                    sub_params = list(pluck(sub_inds, get(ids, params)))
                else:
                    sub_tokens = sub_params = None

                if is_transform:
                    sub_fits, sub_Xs = do_fit_transform(dsk, next_token,
                                                        sub_est, cv, sub_fields,
                                                        sub_tokens, sub_params,
                                                        sub_Xs, sub_ys,
                                                        sub_fit_params,
                                                        n_splits, error_score)
                    new_Xs.update(zip(ids, sub_Xs))
                    new_fits.update(zip(ids, sub_fits))
                else:
                    sub_fits = do_fit(dsk, next_token, sub_est, cv,
                                        sub_fields, sub_tokens, sub_params,
                                        sub_Xs, sub_ys, sub_fit_params,
                                        n_splits, error_score)
                    new_fits.update(zip(ids, sub_fits))
        # Extract lists of transformed Xs and fit steps
        all_ids = list(range(len(Xs)))
        if is_transform:
            Xs = get(all_ids, new_Xs)
        fits = get(all_ids, new_fits)
    elif step is None:
        # Nothing to do
        fits = [None] * len(Xs)
        if not none_passthrough:
            Xs = fits
    else:
        # Only subset the parameters/tokens if necessary
        if sub_fields:
            sub_tokens = list(pluck(sub_inds, tokens))
            sub_params = list(pluck(sub_inds, params))
        else:
            sub_tokens = sub_params = None

        if is_transform:
            fits, Xs = do_fit_transform(dsk, next_token, step, cv,
                                        sub_fields, sub_tokens, sub_params,
                                        Xs, ys, sub_fit_params, n_splits,
                                        error_score)
        else:
            fits = do_fit(dsk, next_token, step, cv, sub_fields,
                            sub_tokens, sub_params, Xs, ys, sub_fit_params,
                            n_splits, error_score)
    return (fits, Xs) if is_transform else (fits, None)


def _do_pipeline(dsk, next_token, est, cv, fields, tokens, params, Xs, ys,
                 fit_params, n_splits, error_score, is_transform):
    if 'steps' in fields:
        raise NotImplementedError("Setting Pipeline.steps in a gridsearch")

    field_to_index, step_fields_lk = _group_subparams(est.steps, fields)
    fit_params_lk = _group_fit_params(est.steps, fit_params)

    # A list of (step, is_transform)
    instrs = [(s, True) for s in est.steps[:-1]]
    instrs.append((est.steps[-1], is_transform))

    fit_steps = []
    for (step_name, step), transform in instrs:
        fits, Xs = _do_fit_step(dsk, next_token, step, cv, fields, tokens,
                                params, Xs, ys, fit_params, n_splits,
                                error_score, step_fields_lk, fit_params_lk,
                                field_to_index, step_name, True, transform)
        fit_steps.append(fits)

    # Rebuild the pipelines
    step_names = [n for n, _ in est.steps]
    out_ests = []
    out_ests_append = out_ests.append
    name = 'pipeline-' + next_token(est)
    m = 0
    seen = {}
    for steps in zip(*fit_steps):
        if steps in seen:
            out_ests_append(seen[steps])
        else:
            for n in range(n_splits):
                dsk[(name, m, n)] = (pipeline, step_names,
                                     [None if s is None else s + (n,)
                                      for s in steps])
            seen[steps] = (name, m)
            out_ests_append((name, m))
            m += 1

    if is_transform:
        return out_ests, Xs
    return out_ests


def _do_n_samples(dsk, token, Xs, n_splits):
    name = 'n_samples-' + token
    n_samples = []
    n_samples_append = n_samples.append
    seen = {}
    m = 0
    for x in Xs:
        if x in seen:
            n_samples_append(seen[x])
        else:
            for n in range(n_splits):
                dsk[name, m, n] = (_num_samples, x + (n,))
            n_samples_append((name, m))
            seen[x] = (name, m)
            m += 1
    return n_samples


def _do_featureunion(dsk, next_token, est, cv, fields, tokens, params, Xs, ys,
                     fit_params, n_splits, error_score):
    if 'transformer_list' in fields:
        raise NotImplementedError("Setting FeatureUnion.transformer_list "
                                  "in a gridsearch")

    (field_to_index,
     step_fields_lk) = _group_subparams(est.transformer_list, fields,
                                        ignore=('transformer_weights'))
    fit_params_lk = _group_fit_params(est.transformer_list, fit_params)

    token = next_token(est)

    n_samples = _do_n_samples(dsk, token, Xs, n_splits)

    fit_steps = []
    tr_Xs = []
    for (step_name, step) in est.transformer_list:
        fits, out_Xs = _do_fit_step(dsk, next_token, step, cv, fields, tokens,
                                    params, Xs, ys, fit_params, n_splits,
                                    error_score, step_fields_lk, fit_params_lk,
                                    field_to_index, step_name, False, True)
        fit_steps.append(fits)
        tr_Xs.append(out_Xs)

    # Rebuild the FeatureUnions
    step_names = [n for n, _ in est.transformer_list]

    if 'transformer_weights' in field_to_index:
        index = field_to_index['transformer_weights']
        weight_lk = {}
        weight_tokens = list(pluck(index, tokens))
        for i, tok in enumerate(weight_tokens):
            if tok not in weight_lk:
                weights = params[i][index]
                if weights is MISSING:
                    weights = est.transformer_weights
                lk = weights or {}
                weight_list = [lk.get(n) for n in step_names]
                weight_lk[tok] = (weights, weight_list)
        weights = get(weight_tokens, weight_lk)
    else:
        lk = est.transformer_weights or {}
        weight_list = [lk.get(n) for n in step_names]
        weight_tokens = repeat(None)
        weights = repeat((est.transformer_weights, weight_list))

    out = []
    out_append = out.append
    fit_name = 'feature-union-' + token
    tr_name = 'feature-union-concat-' + token
    m = 0
    seen = {}
    for steps, Xs, wt, (w, wl), nsamp in zip(zip(*fit_steps), zip(*tr_Xs),
                                             weight_tokens, weights, n_samples):
        if (steps, wt) in seen:
            out_append(seen[steps, wt])
        else:
            for n in range(n_splits):
                dsk[(fit_name, m, n)] = (feature_union, step_names,
                                         [None if s is None else s + (n,)
                                          for s in steps], w)
                dsk[(tr_name, m, n)] = (feature_union_concat,
                                        [None if x is None else x + (n,)
                                         for x in Xs], nsamp + (n,), wl)
            seen[steps, wt] = m
            out_append(m)
            m += 1
    return [(fit_name, i) for i in out], [(tr_name, i) for i in out]


# ------------ #
# CV splitting #
# ------------ #

def check_cv(cv=3, y=None, classifier=False):
    """Dask aware version of ``sklearn.model_selection.check_cv``

    Same as the scikit-learn version, but works if ``y`` is a dask object.
    """
    if cv is None:
        cv = 3

    # If ``cv`` is not an integer, the scikit-learn implementation doesn't
    # touch the ``y`` object, so passing on a dask object is fine
    if not isinstance(y, Base) or not isinstance(cv, numbers.Integral):
        return model_selection.check_cv(cv, y, classifier)

    if classifier:
        # ``y`` is a dask object. We need to compute the target type
        target_type = delayed(type_of_target, pure=True)(y).compute()
        if target_type in ('binary', 'multiclass'):
            return StratifiedKFold(cv)
    return KFold(cv)


def compute_n_splits(cv, X, y=None, groups=None):
    """Return the number of splits.

    Parameters
    ----------
    cv : BaseCrossValidator
    X, y, groups : array_like, dask object, or None

    Returns
    -------
    n_splits : int
    """
    if not any(isinstance(i, Base) for i in (X, y, groups)):
        return cv.get_n_splits(X, y, groups)

    if isinstance(cv, (_BaseKFold, BaseShuffleSplit)):
        return cv.n_splits

    elif isinstance(cv, PredefinedSplit):
        return len(cv.unique_folds)

    elif isinstance(cv, _CVIterableWrapper):
        return len(cv.cv)

    elif isinstance(cv, (LeaveOneOut, LeavePOut)) and not isinstance(X, Base):
        # Only `X` is referenced for these classes
        return cv.get_n_splits(X, None, None)

    elif (isinstance(cv, (LeaveOneGroupOut, LeavePGroupsOut)) and not
          isinstance(groups, Base)):
        # Only `groups` is referenced for these classes
        return cv.get_n_splits(None, None, groups)

    else:
        return delayed(cv).get_n_splits(X, y, groups).compute()


class DaskBaseSearchCV(BaseEstimator, MetaEstimatorMixin):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True, cache_cv=True,
                 get=None):
        self.scoring = scoring
        self.estimator = estimator
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.cache_cv = cache_cv
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

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, shape = [n_samples], optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        error_score = self.error_score
        if not (isinstance(error_score, numbers.Number) or
                error_score == 'raise'):
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value.")

        dsk, keys, n_splits = build_graph(estimator, self.cv, self.scorer_,
                                          list(self._get_param_iterator()),
                                          X, y, groups, fit_params,
                                          iid=self.iid,
                                          refit=self.refit,
                                          error_score=error_score,
                                          return_train_score=self.return_train_score,
                                          cache_cv=self.cache_cv)
        self.dask_graph_ = dsk
        self.n_splits_ = n_splits

        get = self.get or dask.context._globals.get('get') or threaded_get
        out = get(dsk, keys)

        self.cv_results_ = results = out[0]
        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]

        if self.refit:
            self.best_estimator_ = out[1]
        return self

    def visualize(self, filename='mydask', format=None, **kwargs):
        """Render the task graph for this parameter search using ``graphviz``.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional
            The name (without an extension) of the file to write to disk.  If
            `filename` is None, no file will be written, and we communicate
            with dot using only pipes.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file.  Default is 'png'.
        **kwargs
           Additional keyword arguments to forward to ``dask.dot.to_graphviz``.

        Returns
        -------
        result : IPython.diplay.Image, IPython.display.SVG, or None
            See ``dask.dot.dot_graph`` for more information.
        """
        check_is_fitted(self, 'dask_graph_')
        return dask.visualize(self.dask_graph_, filename=filename,
                              format=format, **kwargs)


class GridSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, iid=True,
                 refit=True, cv=None, error_score='raise',
                 return_train_score=True, get=None, cache_cv=True):
        super(GridSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get, cache_cv=cache_cv)

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return model_selection.ParameterGrid(self.param_grid)


class RandomizedSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 iid=True, refit=True, cv=None, random_state=None,
                 error_score='raise', return_train_score=True, get=None,
                 cache_cv=True):
        super(RandomizedSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                get=get, cache_cv=cache_cv)

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                self.n_iter, random_state=self.random_state)
