from __future__ import absolute_import, division, print_function

import numbers
import warnings
from operator import getitem
from collections import defaultdict

import numpy as np
from scipy import sparse

from dask.delayed import delayed
from dask.base import Base, tokenize

from sklearn.base import is_classifier, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import check_cv as _sklearn_check_cv
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
from sklearn.utils import safe_indexing
from sklearn.utils.fixes import rankdata, MaskedArray
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (_num_samples, check_consistent_length,
                                      _is_arraylike)

from ._normalize import normalize_estimator
from .utils import to_indexable, to_keys, copy_estimator, unzip


# A singleton to indicate a failed estimator fit
FIT_FAILURE = type('FitFailure', (object,),
                   {'__slots__': (),
                    '__reduce__': lambda self: 'FIT_FAILURE',
                    '__doc__': "A singleton to indicate fit failure"})()


def warn_fit_failure(error_score, e):
    warnings.warn("Classifier fit failed. The score on this train-test"
                  " partition for these parameters will be set to %f. "
                  "Details: \n%r" % (error_score, e), FitFailedWarning)


# ----------------------- #
# Functions in the graphs #
# ----------------------- #

def cv_split(cv, X, y, groups):
    check_consistent_length(X, y, groups)
    return list(cv.split(X, y, groups))


def cv_extract(X, y, ind):
    return (safe_indexing(X, ind),
            None if y is None else safe_indexing(y, ind))


def cv_extract_pairwise(X, y, ind1, ind2):
    if not hasattr(X, "shape"):
        raise ValueError("Precomputed kernels or affinity matrices have "
                        "to be passed as arrays or sparse matrices.")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X should be a square kernel matrix")
    return (X[np.ix_(ind1, ind2)],
            None if y is None else safe_indexing(y, ind1))


def cv_extract_param(x, indices):
    return safe_indexing(x, indices) if _is_arraylike(x) else x


def pipeline(names, steps):
    """Reconstruct a Pipeline from names and steps"""
    if any(s is FIT_FAILURE for s in steps):
        return FIT_FAILURE
    return Pipeline(list(zip(names, steps)))


def feature_union(names, steps, weights):
    """Reconstruct a FeatureUnion from names, steps, and weights"""
    if any(s is FIT_FAILURE for s in steps):
        return FIT_FAILURE
    return FeatureUnion(list(zip(names, steps)),
                        transformer_weights=weights)


def feature_union_empty(X):
    return np.zeros((X.shape[0], 0))


def feature_union_concat(Xs, weights):
    """Apply weights and concatenate outputs from a FeatureUnion"""
    if any(x is FIT_FAILURE for x in Xs):
        return FIT_FAILURE
    Xs = [X if w is None else X * w for X, w in zip(Xs, weights)]
    if any(sparse.issparse(f) for f in Xs):
        return sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)


def fit(est, X, y, fit_params, error_score='raise'):
    if est is FIT_FAILURE or X is FIT_FAILURE:
        return FIT_FAILURE
    try:
        est = copy_estimator(est)
        est.fit(X, y, **fit_params)
    except Exception as e:
        if error_score == 'raise':
            raise
        warn_fit_failure(error_score, e)
        est = FIT_FAILURE
    return est


def fit_transform(est, X, y, fit_params, error_score='raise'):
    if est is FIT_FAILURE or X is FIT_FAILURE:
        return FIT_FAILURE, FIT_FAILURE
    try:
        est = copy_estimator(est)
        if hasattr(est, 'fit_transform'):
            Xt = est.fit_transform(X, y, **fit_params)
        else:
            est.fit(X, y, **fit_params)
            Xt = est.transform(X)
    except Exception as e:
        if error_score == 'raise':
            raise
        warn_fit_failure(error_score, e)
        est = Xt = FIT_FAILURE
    return est, Xt


def score(est, X, y, scorer):
    if est is FIT_FAILURE:
        return FIT_FAILURE
    return scorer(est, X) if y is None else scorer(est, X, y)


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


def create_cv_results(output, candidate_params, n_splits, error_score,
                      iid, return_train_score):
    if return_train_score:
        train_scores, test_scores, test_sample_counts = unzip(output, 3)
        train_scores = [error_score if s is FIT_FAILURE else s
                        for s in train_scores]
    else:
        test_scores, test_sample_counts = unzip(output, 2)

    test_scores = [error_score if s is FIT_FAILURE else s for s in test_scores]
    # Construct the `cv_results_` dictionary
    results = {'params': candidate_params}
    n_candidates = len(candidate_params)
    test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=int)

    _store(results, 'test_score', test_scores, n_splits, n_candidates,
           splits=True, rank=True,
           weights=test_sample_counts if iid else None)
    if return_train_score:
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
    return results


def get_best_params(candidate_params, cv_results):
    best_index = np.flatnonzero(cv_results["rank_test_score"] == 1)[0]
    return candidate_params[best_index]


def fit_best(estimator, params, X, y, fit_params):
    estimator = copy_estimator(estimator).set_params(**params)
    estimator.fit(X, y, **fit_params)
    return estimator


# -------------- #
# Main Functions #
# -------------- #

def build_graph(estimator, cv, scorer, candidate_params, X, y=None,
                groups=None, fit_params=None, iid=True, refit=True,
                error_score='raise', return_train_score=True):

    X, y, groups = to_indexable(X, y, groups)
    if fit_params:
        param_values = to_indexable(*fit_params.values(), allow_scalars=True)
        fit_params = dict(zip(fit_params, param_values))
    else:
        fit_params = {}
    cv = check_cv(cv, y, is_classifier(estimator))
    # "pairwise" estimators require a different graph for CV splitting
    is_pairwise = getattr(estimator, '_pairwise', False)

    (dsk, n_splits,
     X_name, y_name,
     X_train, y_train,
     X_test, y_test,
     fit_params) = initialize_graph(cv, X, y, groups, fit_params,
                                    is_pairwise)

    scores = []
    for parameters in candidate_params:
        est = clone(estimator).set_params(**parameters)
        score = do_fit_and_score(dsk, est, X_train, y_train, X_test,
                                 y_test, n_splits, scorer,
                                 fit_params=fit_params,
                                 error_score=error_score,
                                 return_train_score=return_train_score)
        scores.extend([[(s, n) for s in score] for n in range(n_splits)])

    token = tokenize(scores)
    cv_results = 'cv-results-' + token
    dsk[cv_results] = (create_cv_results, scores, candidate_params, n_splits,
                       error_score, iid, return_train_score)
    keys = [cv_results]

    if refit:
        best_params = 'best-params-' + token
        dsk[best_params] = (get_best_params, candidate_params, cv_results)
        best_estimator = 'best-estimator-' + token
        dsk[best_estimator] = (fit_best, clone(estimator), best_params,
                               X_name, y_name, _get_fit_params(fit_params, -1))
        keys.append(best_estimator)

    return dsk, keys, n_splits


def do_fit_and_score(dsk, est, X_train, y_train, X_test, y_test, n_splits,
                     scorer, fit_params, error_score, return_train_score):
    # Fit the estimator on the training data
    fit_est = do_fit(dsk, est, X_train, y_train, n_splits, fit_params,
                     error_score)

    test_score = 'test-score-' + tokenize(fit_est, X_test, y_test, scorer)
    n_samples = 'num-samples-' + tokenize(X_test)

    for n in range(n_splits):
        dsk[(test_score, n)] = (score, (fit_est, n), (X_test, n),
                                (y_test, n), scorer)
        dsk[(n_samples, n)] = (_num_samples, (X_test, n))

    if return_train_score:
        train_score = 'train-score-' + tokenize(fit_est, X_train, y_train,
                                                scorer)
        for n in range(n_splits):
            dsk[(train_score, n)] = (score, (fit_est, n), (X_train, n),
                                     (y_train, n), scorer)
        return train_score, test_score, n_samples

    return test_score, n_samples


def do_fit(dsk, est, X, y, n_splits, fit_params, error_score):
    if isinstance(est, Pipeline):
        param_lk = _group_fit_params(est.steps, fit_params)
        fits, Xt = _fit_transform_steps(dsk, est.steps[:-1], X, y, n_splits,
                                        param_lk, error_score)
        name, step = est.steps[-1]
        fits.append(None if step is None else
                    do_fit(dsk, step, Xt, y, n_splits, param_lk[name], error_score))
        return _rebuild_pipeline(dsk, est, fits, n_splits)
    elif isinstance(est, FeatureUnion):
        param_lk = _group_fit_params(est.transformer_list, fit_params)
        fits = [None if tr is None else
                do_fit(dsk, tr, X, y, n_splits, param_lk[name], error_score)
                for name, tr in est.transformer_list]
        return _rebuild_feature_union(dsk, est, fits, n_splits)
    else:
        token = tokenize(normalize_estimator(est), X, y, fit_params,
                         error_score == 'raise')
        name = '%s-fit-%s' % (type(est).__name__.lower(), token)

        for n in range(n_splits):
            dsk[(name, n)] = (fit, est, (X, n), (y, n),
                              _get_fit_params(fit_params, n), error_score)
        return name


def do_fit_transform(dsk, est, X, y, n_splits, fit_params, error_score):
    if isinstance(est, Pipeline):
        param_lk = _group_fit_params(est.steps, fit_params)
        fits, Xt = _fit_transform_steps(dsk, est.steps, X, y, n_splits,
                                        param_lk, error_score)
        fit = _rebuild_pipeline(dsk, est, fits, n_splits)
    elif isinstance(est, FeatureUnion):
        param_lk = _group_fit_params(est.transformer_list, fit_params)
        get_weight = (est.transformer_weights or {}).get
        fits, Xs, weights = [], [], []
        for name, tr in est.transformer_list:
            if tr is None:
                fit_est = None
            else:
                fit_est, Xt = do_fit_transform(dsk, tr, X, y, n_splits,
                                               param_lk[name], error_score)
                Xs.append(Xt)
                weights.append(get_weight(name))
            fits.append(fit_est)
        if not Xs:
            Xt = 'feature-union-transform-' + tokenize(X)
            for n in range(n_splits):
                dsk[(Xt, n)] = (feature_union_empty, (X, n))
        else:
            Xt = 'feature-union-transform-' + tokenize(Xs, weights)
            for n in range(n_splits):
                dsk[(Xt, n)] = (feature_union_concat, [(x, n) for x in Xs],
                                weights)
        fit = _rebuild_feature_union(dsk, est, fits, n_splits)
    else:
        token = tokenize(normalize_estimator(est), X, y, fit_params,
                         error_score == 'raise')
        name = type(est).__name__.lower()
        fit_Xt = '%s-fit-transform-%s' % (name, token)
        fit = '%s-fit-%s' % (name, token)
        Xt = '%s-transform-%s' % (name, token)
        for n in range(n_splits):
            dsk[(fit_Xt, n)] = (fit_transform, est, (X, n), (y, n),
                                _get_fit_params(fit_params, n), error_score)
            dsk[(fit, n)] = (getitem, (fit_Xt, n), 0)
            dsk[(Xt, n)] = (getitem, (fit_Xt, n), 1)
    return fit, Xt


def _group_fit_params(steps, fit_params):
    param_lk = {n: {} for n, s in steps if s is not None}
    for pname, pval in fit_params.items():
        step, param = pname.split('__', 1)
        param_lk[step][param] = pval
    return param_lk


def _get_fit_params(fit_params, n):
    if fit_params:
        return (dict, [[k, (v, n)] for (k, v) in fit_params.items()])
    return fit_params


def _fit_transform_steps(dsk, steps, Xt, y, n_splits, param_lk, error_score):
    fits = []
    for name, step in steps:
        if step is None:
            fit_est = None
        else:
            fit_est, Xt = do_fit_transform(dsk, step, Xt, y, n_splits,
                                           param_lk[name], error_score)
        fits.append(fit_est)
    return fits, Xt


def _rebuild_pipeline(dsk, est, fits, n_splits):
    names = [n for n, _ in est.steps]
    name = 'pipeline-' + tokenize(fits, names)
    for n in range(n_splits):
        dsk[(name, n)] = (pipeline, names,
                          [f if f is None else (f, n) for f in fits])
    return name


def _rebuild_feature_union(dsk, est, fits, n_splits):
    names = [n for n, _ in est.transformer_list]
    name = tokenize(fits, names, est.transformer_weights)
    for n in range(n_splits):
        dsk[(name, n)] = (feature_union, names,
                          [f if f is None else (f, n) for f in fits],
                          est.transformer_weights)
    return name


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
        return _sklearn_check_cv(cv, y, classifier)

    if classifier:
        # ``y`` is a dask object. We need to compute the target type
        target_type = delayed(type_of_target, pure=True)(y).compute()
        if target_type in ('binary', 'multiclass'):
            return StratifiedKFold(cv)
    return KFold(cv)


def initialize_graph(cv, X, y=None, groups=None, fit_params=None,
                     is_pairwise=False):
    """Initialize the CV dask graph.

    Given input data X, y, and groups, build up a dask graph performing the
    initial CV splits.

    Parameters
    ----------
    cv : BaseCrossValidator
    X, y, groups : array_like, dask object, or None
    fit_params : dict, optional
    is_pairwise : bool
        Whether the estimator being evaluated has ``_pairwise`` as an
        attribute (which affects how the CV splitting is done).
    """
    dsk = {}
    X_name, y_name, groups_name = to_keys(dsk, X, y, groups)
    n_splits = compute_n_splits(cv, X, y, groups)

    if not fit_params:
        fit_params = {}

    cv_token = tokenize(cv, X_name, y_name, groups_name, is_pairwise)
    cv_name = 'cv-split-' + cv_token
    train_name = 'cv-split-train-' + cv_token
    test_name = 'cv-split-test-' + cv_token

    dsk[cv_name] = (cv_split, cv, X_name, y_name, groups_name)

    for n in range(n_splits):
        dsk[(cv_name, n)] = (getitem, cv_name, n)
        dsk[(train_name, n)] = (getitem, (cv_name, n), 0)
        dsk[(test_name, n)] = (getitem, (cv_name, n), 1)

    # Extract the test-train subsets
    Xy_train = 'Xy-train-' + cv_token
    X_train = 'X-train-' + cv_token
    y_train = 'y-train-' + cv_token
    Xy_test = 'Xy-test-' + cv_token
    X_test = 'X-test-' + cv_token
    y_test = 'y-test-' + cv_token

    # Build a helper function to insert the extract tasks
    if is_pairwise:
        def extract_train(X, y, train, test):
            return (cv_extract_pairwise, X, y, train, train)

        def extract_test(X, y, train, test):
            return (cv_extract_pairwise, X, y, test, train)
    else:
        def extract_train(X, y, train, test):
            return (cv_extract, X, y, train)

        def extract_test(X, y, train, test):
            return (cv_extract, X, y, test)

    for n in range(n_splits):
        dsk[(Xy_train, n)] = extract_train(X_name, y_name, (train_name, n),
                                           (test_name, n))
        dsk[(X_train, n)] = (getitem, (Xy_train, n), 0)
        dsk[(y_train, n)] = (getitem, (Xy_train, n), 1)
        dsk[(Xy_test, n)] = extract_test(X_name, y_name, (train_name, n),
                                         (test_name, n))
        dsk[(X_test, n)] = (getitem, (Xy_test, n), 0)
        dsk[(y_test, n)] = (getitem, (Xy_test, n), 1)

    fit_params_train = {}
    if fit_params:
        for name, val in zip(fit_params, to_keys(dsk, *fit_params.values())):
            param_key = 'fit-param-train' + tokenize(name, val, cv_token)
            dsk[(param_key, -1)] = val
            for n in range(n_splits):
                dsk[(param_key, n)] = (cv_extract_param, (param_key, -1),
                                       (train_name, n))
            fit_params_train[name] = param_key

    return (dsk, n_splits,
            X_name, y_name,
            X_train, y_train,
            X_test, y_test,
            fit_params_train)


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
