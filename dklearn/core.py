from __future__ import absolute_import, division, print_function

from operator import getitem

import numpy as np
from scipy import sparse

from dask.base import tokenize
from sklearn.base import clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import safe_indexing
from sklearn.utils.validation import _num_samples

from . import normalize  # Need to import to register normalize methods
del normalize


# ----------------------- #
# Functions in the graphs #
# ----------------------- #

def pipeline(names, steps):
    """Reconstruct a Pipeline from names and steps"""
    return Pipeline(list(zip(names, steps)))


def feature_union(names, steps, weights):
    """Reconstruct a FeatureUnion from names, steps, and weights"""
    return FeatureUnion(list(zip(names, steps)),
                        transformer_weights=weights)


def feature_union_empty(X):
    return np.zeros((X.shape[0], 0))


def feature_union_concat(Xs, weights):
    """Apply weights and concatenate outputs from a FeatureUnion"""
    Xs = [X if w is None else X * w for X, w in zip(Xs, weights)]
    if any(sparse.issparse(f) for f in Xs):
        return sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)


def fit(est, X, y, fit_params):
    return clone(est).fit(X, y, **fit_params)


def fit_transform(est, X, y, fit_params):
    est = clone(est)
    if hasattr(est, 'fit_transform'):
        Xt = est.fit_transform(X, y, **fit_params)
    else:
        est.fit(X, y, **fit_params)
        Xt = est.transform(X)
    return est, Xt


def score(est, X, y, scorer):
    return scorer(est, X) if y is None else scorer(est, X, y)


def cv_split(cv, X, y, groups):
    return list(cv.split(X, y, groups))


def cv_extract(x, splits, index):
    return None if x is None else safe_indexing(x, splits[index])


# -------------- #
# Main Functions #
# -------------- #


def initialize_dask_graph(X, y, cv, groups):
    """Initialize a dask graph and key names for a CV run.

    Parameters
    ----------
    X, y : array_like
    cv : BaseCrossValidator
    groups : array_like
    """
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
    X_train = 'X-train-' + cv_token
    y_train = 'y-train-' + cv_token
    X_test = 'X-test-' + cv_token
    y_test = 'y-test-' + cv_token
    for n in range(n_splits):
        dsk[(X_train, n)] = (cv_extract, X_name, (cv_name, n), 0)
        dsk[(y_train, n)] = (cv_extract, y_name, (cv_name, n), 0)
        dsk[(X_test, n)] = (cv_extract, X_name, (cv_name, n), 1)
        dsk[(y_test, n)] = (cv_extract, y_name, (cv_name, n), 1)

    return dsk, X_train, y_train, X_test, y_test, n_splits


def do_fit_and_score(dsk, est, X_train, y_train, X_test, y_test, n_splits,
                     scorer, fit_params, return_train_score, error_score):
    # Fit the estimator on the training data
    fit_est = do_fit(dsk, est, X_train, y_train, n_splits, fit_params)

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


def do_fit(dsk, est, X, y, n_splits, fit_params):
    if isinstance(est, Pipeline):
        func = do_fit_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_feature_union
    else:
        func = do_fit_estimator
    return func(dsk, est, X, y, n_splits, fit_params)


def do_fit_transform(dsk, est, X, y, n_splits, fit_params):
    if isinstance(est, Pipeline):
        func = do_fit_transform_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_transform_feature_union
    else:
        func = do_fit_transform_estimator
    return func(dsk, est, X, y, n_splits, fit_params)


# --------- #
# Estimator #
# --------- #

def do_fit_estimator(dsk, est, X, y, n_splits, fit_params):
    token = tokenize(est, X, y, fit_params)
    est_name = type(est).__name__.lower()
    name = '%s-fit-%s' % (est_name, token)
    for n in range(n_splits):
        dsk[(name, n)] = (fit, est, (X, n), (y, n), fit_params)
    return name


def do_fit_transform_estimator(dsk, est, X, y, n_splits, fit_params):
    token = tokenize(est, X, y, fit_params)
    name = type(est).__name__.lower()
    fit_tr_name = '%s-fit-transform-%s' % (name, token)
    fit_name = '%s-fit-%s' % (name, token)
    tr_name = '%s-transform-%s' % (name, token)
    for n in range(n_splits):
        dsk[(fit_tr_name, n)] = (fit_transform, est, (X, n), (y, n), fit_params)
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


def _fit_transform_steps(dsk, steps, Xt, y, n_splits, param_lk):
    fits = []
    for name, step in steps:
        if step is None:
            fit_est = None
        else:
            fit_est, Xt = do_fit_transform(dsk, step, Xt, y, n_splits,
                                           param_lk[name])
        fits.append(fit_est)
    return fits, Xt


def _rebuild_pipeline(dsk, est, fits, n_splits):
    names = [n for n, _ in est.steps]
    name = 'pipeline-' + tokenize(fits, names)
    for n in range(n_splits):
        dsk[(name, n)] = (pipeline, names,
                          [f if f is None else (f, n) for f in fits])
    return name


def do_fit_transform_pipeline(dsk, est, X, y, n_splits, fit_params):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps, X, y, n_splits, param_lk)
    return _rebuild_pipeline(dsk, est, fits, n_splits), Xt


def do_fit_pipeline(dsk, est, X, y, n_splits, fit_params):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps[:-1], X, y, n_splits,
                                    param_lk)
    name, step = est.steps[-1]
    fits.append(None if step is None
                else do_fit(dsk, step, Xt, y, n_splits, param_lk[name]))
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


def do_fit_transform_feature_union(dsk, est, X, y, n_splits, fit_params):
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
                                           param_lk[name])
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


def do_fit_feature_union(dsk, est, X, y, n_splits, fit_params):
    param_lk = _group_fit_params(est.transformer_list, fit_params)
    fits = [None if tr is None
            else do_fit(dsk, tr, X, y, n_splits, param_lk[name])
            for name, tr in est.transformer_list]
    return _rebuild_feature_union(dsk, est, fits, n_splits)
