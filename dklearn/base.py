from __future__ import absolute_import, division, print_function

from operator import getitem

import numpy as np
from scipy import sparse

from dask.base import tokenize, normalize_token
from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion


@normalize_token.register(BaseEstimator)
def normalize_BaseEstimator(est):
    return type(est).__name__, normalize_token(est.get_params())


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


# -------------- #
# Main Functions #
# -------------- #


def do_fit(dsk, est, X, y, fit_params):
    if isinstance(est, Pipeline):
        func = do_fit_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_feature_union
    else:
        func = do_fit_estimator
    return func(dsk, est, X, y, fit_params)


def do_fit_transform(dsk, est, X, y, fit_params):
    if isinstance(est, Pipeline):
        func = do_fit_transform_pipeline
    elif isinstance(est, FeatureUnion):
        func = do_fit_transform_feature_union
    else:
        func = do_fit_transform_estimator
    return func(dsk, est, X, y, fit_params)


# --------- #
# Estimator #
# --------- #

def do_fit_estimator(dsk, est, X, y, fit_params):
    token = tokenize(est, X, y, fit_params)
    est_name = type(est).__name__.lower()
    name = '%s-fit-%s' % (est_name, token)
    dsk[name] = (fit, est, X, y, fit_params)
    return name


def do_fit_transform_estimator(dsk, est, X, y, fit_params):
    token = tokenize(est, X, y, fit_params)
    name = type(est).__name__.lower()
    fit_tr_name = '%s-fit-transform-%s' % (name, token)
    fit_name = '%s-fit-%s' % (name, token)
    tr_name = '%s-transform-%s' % (name, token)
    dsk[fit_tr_name] = (fit_transform, est, X, y, fit_params)
    dsk[fit_name] = (getitem, fit_tr_name, 0)
    dsk[tr_name] = (getitem, fit_tr_name, 1)
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


def _fit_transform_steps(dsk, steps, Xt, y, param_lk):
    fits = []
    for name, step in steps:
        if step is None:
            fit_est = None
        else:
            fit_est, Xt = do_fit_transform(dsk, step, Xt, y, param_lk[name])
        fits.append(fit_est)
    return fits, Xt


def _rebuild_pipeline(dsk, est, fits):
    names = [n for n, _ in est.steps]
    name = 'pipeline-' + tokenize(fits, names)
    dsk[name] = (pipeline, names, fits)
    return name


def do_fit_transform_pipeline(dsk, est, X, y, fit_params):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps, X, y, param_lk)
    return _rebuild_pipeline(dsk, est, fits), Xt


def do_fit_pipeline(dsk, est, X, y, fit_params):
    param_lk = _group_fit_params(est.steps, fit_params)
    fits, Xt = _fit_transform_steps(dsk, est.steps[:-1], X, y, param_lk)
    name, step = est.steps[-1]
    fits.append(None if step is None
                else do_fit(dsk, step, Xt, y, param_lk[name]))
    return _rebuild_pipeline(dsk, est, fits)


# ------------ #
# FeatureUnion #
# ------------ #

def _rebuild_feature_union(dsk, est, fits):
    names = [n for n, _ in est.transformer_list]
    fit = tokenize(fits, names, est.transformer_weights)
    dsk[fit] = (feature_union, fits, names, est.transformer_weights)
    return fit


def do_fit_transform_feature_union(dsk, est, X, y, fit_params):
    param_lk = _group_fit_params(est.transformer_list, fit_params)
    get_weight = (est.transformer_weights or {}).get
    fits = []
    Xs = []
    weights = []
    for name, tr in est.transformer_list:
        if tr is None:
            fit_est = None
        else:
            fit_est, Xt = do_fit_transform(dsk, tr, X, y, param_lk[name])
            Xs.append(Xt)
            weights.append(get_weight(name))
        fits.append(fit_est)

    if not Xs:
        Xt = 'feature-union-transform-' + tokenize(X)
        dsk[Xt] = (feature_union_empty, X)
    else:
        Xt = 'feature-union-transform-' + tokenize(Xs, weights)
        dsk[Xt] = (feature_union_concat, Xs, weights)

    return _rebuild_feature_union(dsk, est, fits), Xt


def do_fit_feature_union(dsk, est, X, y, fit_params):
    param_lk = _group_fit_params(est.transformer_list, fit_params)
    fits = [None if tr is None else do_fit(dsk, tr, X, y, param_lk[name])
            for name, tr in est.transformer_list]
    return _rebuild_feature_union(dsk, est, fits)
