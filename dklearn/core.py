from __future__ import absolute_import, division, print_function

from operator import getitem
from collections import defaultdict
from itertools import repeat
import numbers

from dask.base import tokenize, Base
from dask.delayed import delayed
from sklearn.base import is_classifier, clone
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
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from .methods import (fit, fit_transform, fit_and_score, pipeline, fit_best,
                      get_best_params, create_cv_results, cv_split,
                      cv_n_samples, cv_extract, cv_extract_params,
                      decompress_params, score, MISSING)
from ._normalize import normalize_estimator

from .utils import to_indexable, to_keys, unzip


try:
    from cytoolz import get, pluck
except:  # pragma: no cover
    from toolz import get, pluck


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


def _do_pipeline(dsk, next_token, est, cv, fields, tokens, params, Xs, ys,
                 fit_params, n_splits, error_score, is_transform):
    if 'steps' in fields:
        raise NotImplementedError("Setting Pipeline.steps in a gridsearch")

    # Group the fields into a mapping of {stepname: [(newname, orig_index)]}
    field_to_index = dict(zip(fields, range(len(fields))))
    step_fields_lk = {s: [] for s, _ in est.steps}
    for f in fields:
        if '__' in f:
            step, param = f.split('__', 1)
            if step in step_fields_lk:
                step_fields_lk[step].append((param, field_to_index[f]))
                continue
        if f not in step_fields_lk:
            raise ValueError("Unknown parameter: `%s`" % f)

    fit_params_lk = _group_fit_params(est.steps, fit_params)

    # A list of (step, is_transform)
    instrs = [(s, True) for s in est.steps[:-1]]
    instrs.append((est.steps[-1], is_transform))

    fit_steps = []
    for (step_name, step), transform in instrs:
        sub_fields, sub_inds = map(list, unzip(step_fields_lk[step_name], 2))

        sub_fit_params = fit_params_lk[step_name]

        if step_name in field_to_index:
            # The estimator may change each call
            new_fits = {}
            new_Xs = {}
            est_index = field_to_index[step_name]

            id_groups = []

            def new_group():
                o = []
                id_groups.append(o)
                return o.append

            _id_groups = defaultdict(new_group)
            for n, step_token in enumerate(pluck(est_index, tokens)):
                _id_groups[step_token](n)

            for ids in id_groups:
                # Get the estimator for this subgroup
                sub_est = params[ids[0]][est_index]
                if sub_est is MISSING:
                    sub_est = step

                # If an estimator is `None`, there's nothing to do
                if sub_est is None:
                    new_fits.update(dict.fromkeys(ids, None))
                    if transform:
                        new_Xs.update(zip(ids, get(ids, Xs)))
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

                    if transform:
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
            if transform:
                Xs = get(all_ids, new_Xs)
            fits = get(all_ids, new_fits)
        elif step is None:
            # Nothing to do
            fits = [None] * len(Xs)
        else:
            # Only subset the parameters/tokens if necessary
            if sub_fields:
                sub_tokens = list(pluck(sub_inds, tokens))
                sub_params = list(pluck(sub_inds, params))
            else:
                sub_tokens = sub_params = None

            if transform:
                fits, Xs = do_fit_transform(dsk, next_token, step, cv,
                                            sub_fields, sub_tokens, sub_params,
                                            Xs, ys, sub_fit_params, n_splits,
                                            error_score)
            else:
                fits = do_fit(dsk, next_token, step, cv, sub_fields,
                              sub_tokens, sub_params, Xs, ys, sub_fit_params,
                              n_splits, error_score)
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
