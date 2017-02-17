from __future__ import absolute_import, division, print_function

from operator import getitem
import warnings
import copy

import numpy as np
from scipy import sparse

from dask.base import tokenize
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import safe_indexing
from sklearn.utils.validation import _num_samples

from .normalize import normalize_estimator


# A singleton to indicate a failed estimator fit
FIT_FAILURE = type('FitFailure', (object,),
                   {'__slots__': (),
                    '__reduce__': lambda self: 'FIT_FAILURE',
                    '__doc__': "A singleton to indicate fit failure"})()


def warn_fit_failure(error_score, e):
    warnings.warn("Classifier fit failed. The score on this train-test"
                  " partition for these parameters will be set to %f. "
                  "Details: \n%r" % (error_score, e), FitFailedWarning)


def fit_failure_to_error_score(scores, error_score):
    return [error_score if s is FIT_FAILURE else s for s in scores]


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)


# ----------------------- #
# Functions in the graphs #
# ----------------------- #

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


def cv_split(cv, X, y, groups):
    return list(cv.split(X, y, groups))


def cv_extract(X, y, splits, is_pairwise=False, is_train=True):
    if is_train:
        indices = indices2 = splits[0]
    else:
        indices2, indices = splits
    if is_pairwise:
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                            "to be passed as arrays or sparse matrices.")
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        X_subset = X[np.ix_(indices, indices2)]
    else:
        X_subset = safe_indexing(X, indices)

    y_subset = None if y is None else safe_indexing(y, indices)

    return X_subset, y_subset


# -------------- #
# Main Functions #
# -------------- #

def initialize_dask_graph(estimator, X, y, cv, groups):
    """Initialize a dask graph and key names for a CV run.

    Parameters
    ----------
    estimator
    X, y : array_like
    cv : BaseCrossValidator
    groups : array_like
    """
    is_pairwise = getattr(estimator, '_pairwise', False)
    n_splits = cv.get_n_splits(X, y, groups)

    X_name = 'X-' + tokenize(X)
    y_name = 'y-' + tokenize(y)
    dsk = {X_name: X, y_name: y}

    # TODO: for certain CrossValidator classes, should be able to generate the
    # `nth` split individually, removing the single task bottleneck we
    # currently have.
    cv_token = tokenize(cv, X_name, y_name, groups)
    cv_name = 'cv-split-' + cv_token
    dsk[cv_name] = (cv_split, cv, X_name, y_name, groups)
    dsk.update(((cv_name, n), (getitem, cv_name, n)) for n in range(n_splits))

    # Extract the test-train subsets
    Xy_train = 'Xy-train-' + cv_token
    X_train = 'X-train-' + cv_token
    y_train = 'y-train-' + cv_token
    Xy_test = 'Xy-test-' + cv_token
    X_test = 'X-test-' + cv_token
    y_test = 'y-test-' + cv_token
    for n in range(n_splits):
        dsk[(Xy_train, n)] = (cv_extract, X_name, y_name, (cv_name, n),
                              is_pairwise, True)
        dsk[(X_train, n)] = (getitem, (Xy_train, n), 0)
        dsk[(y_train, n)] = (getitem, (Xy_train, n), 1)
        dsk[(Xy_test, n)] = (cv_extract, X_name, y_name, (cv_name, n),
                             is_pairwise, False)
        dsk[(X_test, n)] = (getitem, (Xy_test, n), 0)
        dsk[(y_test, n)] = (getitem, (Xy_test, n), 1)

    return dsk, X_train, y_train, X_test, y_test, n_splits


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
        func = do_fit_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_feature_union
    else:
        func = do_fit_estimator
    return func(dsk, est, X, y, n_splits, fit_params, error_score)


def do_fit_transform(dsk, est, X, y, n_splits, fit_params, error_score):
    if isinstance(est, Pipeline):
        func = do_fit_transform_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_transform_feature_union
    else:
        func = do_fit_transform_estimator
    return func(dsk, est, X, y, n_splits, fit_params, error_score)


# --------- #
# Estimator #
# --------- #

def do_fit_estimator(dsk, est, X, y, n_splits, fit_params, error_score):
    token = tokenize(normalize_estimator(est), X, y, fit_params,
                     error_score == 'raise')
    est_name = type(est).__name__.lower()
    name = '%s-fit-%s' % (est_name, token)
    for n in range(n_splits):
        dsk[(name, n)] = (fit, est, (X, n), (y, n), fit_params, error_score)
    return name


def do_fit_transform_estimator(dsk, est, X, y, n_splits, fit_params,
                               error_score):
    token = tokenize(normalize_estimator(est), X, y, fit_params,
                     error_score == 'raise')
    name = type(est).__name__.lower()
    fit_tr_name = '%s-fit-transform-%s' % (name, token)
    fit_name = '%s-fit-%s' % (name, token)
    tr_name = '%s-transform-%s' % (name, token)
    for n in range(n_splits):
        dsk[(fit_tr_name, n)] = (fit_transform, est, (X, n), (y, n),
                                 fit_params, error_score)
        dsk[(fit_name, n)] = (getitem, (fit_tr_name, n), 0)
        dsk[(tr_name, n)] = (getitem, (fit_tr_name, n), 1)
    return fit_name, tr_name


# -------- #
# Pipeline #
# -------- #

def _group_fit_params(steps, fit_params):
    param_lk = {n: {} for n, s in steps if s is not None}
    for pname, pval in fit_params.items():
        step, param = pname.split('__', 1)
        param_lk[step][param] = pval
    return param_lk


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


def do_fit_transform_pipeline(dsk, est, X, y, n_splits, fit_params,
                              error_score):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps, X, y, n_splits, param_lk)
    return _rebuild_pipeline(dsk, est, fits, n_splits), Xt


def do_fit_pipeline(dsk, est, X, y, n_splits, fit_params, error_score):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps[:-1], X, y, n_splits,
                                    param_lk, error_score)
    name, step = est.steps[-1]
    fits.append(None if step is None
                else do_fit(dsk, step, Xt, y, n_splits,
                            param_lk[name], error_score))
    return _rebuild_pipeline(dsk, est, fits, n_splits)


# ------------ #
# FeatureUnion #
# ------------ #

def _rebuild_feature_union(dsk, est, fits, n_splits):
    names = [n for n, _ in est.transformer_list]
    name = tokenize(fits, names, est.transformer_weights)
    for n in range(n_splits):
        dsk[(name, n)] = (feature_union, names,
                          [f if f is None else (f, n) for f in fits],
                          est.transformer_weights)
    return name


def do_fit_transform_feature_union(dsk, est, X, y, n_splits, fit_params,
                                   error_score):
    param_lk = _group_fit_params(est.transformer_list, fit_params)
    get_weight = (est.transformer_weights or {}).get
    fits = []
    Xs = []
    weights = []
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
            dsk[(Xt, n)] = (feature_union_concat, [(x, n) for x in Xs], weights)

    return _rebuild_feature_union(dsk, est, fits, n_splits), Xt


def do_fit_feature_union(dsk, est, X, y, n_splits, fit_params, error_score):
    param_lk = _group_fit_params(est.transformer_list, fit_params)
    fits = [None if tr is None
            else do_fit(dsk, tr, X, y, n_splits, param_lk[name], error_score)
            for name, tr in est.transformer_list]
    return _rebuild_feature_union(dsk, est, fits, n_splits)
