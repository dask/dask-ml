from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from threading import Lock
from timeit import default_timer

import numpy as np
from dask.base import normalize_token
from scipy import sparse
from scipy.stats import rankdata
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_consistent_length
from toolz import pluck

from .._compat import Mapping
from .utils import _index_param_value, _num_samples, _safe_indexing, copy_estimator

# Copied from scikit-learn/sklearn/utils/fixes.py, can be removed once we drop
# support for scikit-learn < 0.18.1 or numpy < 1.12.0.
if LooseVersion(np.__version__) < "1.12.0":

    class MaskedArray(np.ma.MaskedArray):
        # Before numpy 1.12, np.ma.MaskedArray object is not picklable
        # This fix is needed to make our model_selection.GridSearchCV
        # picklable as the ``cv_results_`` param uses MaskedArray
        def __getstate__(self):
            """Return the internal state of the masked array, for pickling
            purposes.

            """
            cf = "CF"[self.flags.fnc]
            data_state = super(np.ma.MaskedArray, self).__reduce__()[2]
            return data_state + (
                np.ma.getmaskarray(self).tostring(cf),
                self._fill_value,
            )


else:
    from numpy.ma import MaskedArray  # noqa

# A singleton to indicate a missing parameter
MISSING = type(
    "MissingParameter",
    (object,),
    {
        "__slots__": (),
        "__reduce__": lambda self: "MISSING",
        "__doc__": "A singleton to indicate a missing parameter",
    },
)()
normalize_token.register(type(MISSING), lambda x: "MISSING")


# A singleton to indicate a failed estimator fit
FIT_FAILURE = type(
    "FitFailure",
    (object,),
    {
        "__slots__": (),
        "__reduce__": lambda self: "FIT_FAILURE",
        "__doc__": "A singleton to indicate fit failure",
    },
)()


def warn_fit_failure(error_score, e):
    warnings.warn(
        "Classifier fit failed. The score on this train-test"
        " partition for these parameters will be set to %f. "
        "Details: \n%r" % (error_score, e),
        FitFailedWarning,
    )


# ----------------------- #
# Functions in the graphs #
# ----------------------- #


class CVCache:
    def __init__(self, splits, pairwise=False, cache=True, num_train_samples=None):
        self.splits = splits
        self.pairwise = pairwise
        self.cache = {} if cache else None
        self.num_train_samples = num_train_samples

    def __reduce__(self):
        return (
            CVCache,
            (
                self.splits,
                self.pairwise,
                self.cache is not None,
                self.num_train_samples,
            ),
        )

    def num_test_samples(self):
        return np.array(
            [i.sum() if i.dtype == bool else len(i) for i in pluck(1, self.splits)]
        )

    def extract(self, X, y, n, is_x=True, is_train=True):
        if is_x:
            if self.pairwise:
                return self._extract_pairwise(X, y, n, is_train=is_train)
            return self._extract(X, y, n, is_x=True, is_train=is_train)
        if y is None:
            return None
        return self._extract(X, y, n, is_x=False, is_train=is_train)

    def extract_param(self, key, x, n):
        if self.cache is not None and (n, key) in self.cache:
            return self.cache[n, key]

        out = _index_param_value(self.num_train_samples, x, self.splits[n][0])

        if self.cache is not None:
            self.cache[n, key] = out
        return out

    def _extract(self, X, y, n, is_x=True, is_train=True):
        if self.cache is not None and (n, is_x, is_train) in self.cache:
            return self.cache[n, is_x, is_train]

        inds = self.splits[n][0] if is_train else self.splits[n][1]
        result = _safe_indexing(X if is_x else y, inds)

        if self.cache is not None:
            self.cache[n, is_x, is_train] = result
        return result

    def _extract_pairwise(self, X, y, n, is_train=True):
        if self.cache is not None and (n, True, is_train) in self.cache:
            return self.cache[n, True, is_train]

        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have "
                "to be passed as arrays or sparse matrices."
            )
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        train, test = self.splits[n]
        result = X[np.ix_(train if is_train else test, train)]

        if self.cache is not None:
            self.cache[n, True, is_train] = result
        return result


def cv_split(cv, X, y, groups, is_pairwise, cache):
    check_consistent_length(X, y, groups)
    return CVCache(list(cv.split(X, y, groups)), is_pairwise, cache, _num_samples(X))


def cv_n_samples(cvs):
    return cvs.num_test_samples()


def cv_extract(cvs, X, y, is_X, is_train, n):
    return cvs.extract(X, y, n, is_X, is_train)


def cv_extract_params(cvs, keys, vals, n):
    return {k: cvs.extract_param(tok, v, n) for (k, tok), v in zip(keys, vals)}


def _maybe_timed(x):
    """Unpack (est, fit_time) tuples if provided"""
    return x if isinstance(x, tuple) and len(x) == 2 else (x, 0.0)


def pipeline(names, steps):
    """Reconstruct a Pipeline from names and steps"""
    steps, times = zip(*map(_maybe_timed, steps))
    fit_time = sum(times)
    if any(s is FIT_FAILURE for s in steps):
        fit_est = FIT_FAILURE
    else:
        fit_est = Pipeline(list(zip(names, steps)))
    return fit_est, fit_time


def feature_union(names, steps, weights):
    """Reconstruct a FeatureUnion from names, steps, and weights"""
    steps, times = zip(*map(_maybe_timed, steps))
    fit_time = sum(times)
    if any(s is FIT_FAILURE for s in steps):
        fit_est = FIT_FAILURE
    else:
        fit_est = FeatureUnion(list(zip(names, steps)), transformer_weights=weights)
    return fit_est, fit_time


def feature_union_concat(Xs, nsamples, weights):
    """Apply weights and concatenate outputs from a FeatureUnion"""
    if any(x is FIT_FAILURE for x in Xs):
        return FIT_FAILURE
    Xs = [X if w is None else X * w for X, w in zip(Xs, weights) if X is not None]
    if not Xs:
        return np.zeros((nsamples, 0))
    if any(sparse.issparse(f) for f in Xs):
        return sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)


# Current set_params isn't threadsafe
SET_PARAMS_LOCK = Lock()


def set_params(est, fields=None, params=None, copy=True):
    if copy:
        est = copy_estimator(est)
    if fields is None:
        return est
    params = {f: p for (f, p) in zip(fields, params) if p is not MISSING}
    # TODO: rewrite set_params to avoid lock for classes that use the standard
    # set_params/get_params methods
    with SET_PARAMS_LOCK:
        return est.set_params(**params)


def fit(est, X, y, error_score="raise", fields=None, params=None, fit_params=None):
    if X is FIT_FAILURE:
        est, fit_time = FIT_FAILURE, 0.0
    else:
        if not fit_params:
            fit_params = {}
        start_time = default_timer()
        try:
            est = set_params(est, fields, params)
            est.fit(X, y, **fit_params)
        except Exception as e:
            if error_score == "raise":
                raise
            warn_fit_failure(error_score, e)
            est = FIT_FAILURE
        fit_time = default_timer() - start_time

    return est, fit_time


def fit_transform(
    est, X, y, error_score="raise", fields=None, params=None, fit_params=None
):
    if X is FIT_FAILURE:
        est, fit_time, Xt = FIT_FAILURE, 0.0, FIT_FAILURE
    else:
        if not fit_params:
            fit_params = {}
        start_time = default_timer()
        try:
            est = set_params(est, fields, params)
            if hasattr(est, "fit_transform"):
                Xt = est.fit_transform(X, y, **fit_params)
            else:
                est.fit(X, y, **fit_params)
                Xt = est.transform(X)
        except Exception as e:
            if error_score == "raise":
                raise
            warn_fit_failure(error_score, e)
            est = Xt = FIT_FAILURE
        fit_time = default_timer() - start_time

    return (est, fit_time), Xt


def _score(est, X, y, scorer):
    if est is FIT_FAILURE:
        return FIT_FAILURE
    if isinstance(scorer, Mapping):
        return {k: v(est, X) if y is None else v(est, X, y) for k, v in scorer.items()}
    return scorer(est, X) if y is None else scorer(est, X, y)


def score(est_and_time, X_test, y_test, X_train, y_train, scorer, error_score):
    est, fit_time = est_and_time
    start_time = default_timer()
    try:
        test_score = _score(est, X_test, y_test, scorer)
    except Exception:
        if error_score == "raise":
            raise
        else:
            score_time = default_timer() - start_time
            return fit_time, error_score, score_time, error_score
    score_time = default_timer() - start_time
    if X_train is None:
        return fit_time, test_score, score_time
    train_score = _score(est, X_train, y_train, scorer)
    return fit_time, test_score, score_time, train_score


def fit_and_score(
    est,
    cv,
    X,
    y,
    n,
    scorer,
    error_score="raise",
    fields=None,
    params=None,
    fit_params=None,
    return_train_score=True,
):
    X_train = cv.extract(X, y, n, True, True)
    y_train = cv.extract(X, y, n, False, True)
    X_test = cv.extract(X, y, n, True, False)
    y_test = cv.extract(X, y, n, False, False)
    est_and_time = fit(est, X_train, y_train, error_score, fields, params, fit_params)
    if not return_train_score:
        X_train = y_train = None

    return score(est_and_time, X_test, y_test, X_train, y_train, scorer, error_score)


def _store(
    results,
    key_name,
    array,
    n_splits,
    n_candidates,
    weights=None,
    splits=False,
    rank=False,
):
    """A small helper to store the scores/times to the cv_results_"""
    # When iterated first by n_splits and then by parameters
    array = np.array(array, dtype=np.float64).reshape(n_splits, n_candidates).T
    if splits:
        for split_i in range(n_splits):
            results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

    array_means = np.average(array, axis=1, weights=weights)
    results["mean_%s" % key_name] = array_means
    # Weighted std is not directly available in numpy
    array_stds = np.sqrt(
        np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights)
    )
    results["std_%s" % key_name] = array_stds

    if rank:
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-array_means, method="min"), dtype=np.int32
        )


def create_cv_results(
    scores, candidate_params, n_splits, error_score, weights, multimetric
):
    if len(scores[0]) == 4:
        fit_times, test_scores, score_times, train_scores = zip(*scores)
    else:
        fit_times, test_scores, score_times = zip(*scores)
        train_scores = None

    if not multimetric:
        test_scores = [error_score if s is FIT_FAILURE else s for s in test_scores]
        if train_scores is not None:
            train_scores = [
                error_score if s is FIT_FAILURE else s for s in train_scores
            ]
    else:
        test_scores = {
            k: [error_score if x is FIT_FAILURE else x[k] for x in test_scores]
            for k in multimetric
        }
        if train_scores is not None:
            train_scores = {
                k: [error_score if x is FIT_FAILURE else x[k] for x in train_scores]
                for k in multimetric
            }

    # Construct the `cv_results_` dictionary
    results = {"params": candidate_params}
    n_candidates = len(candidate_params)

    if weights is not None:
        weights = np.broadcast_to(
            weights[None, :], (len(candidate_params), len(weights))
        )

    _store(results, "fit_time", fit_times, n_splits, n_candidates)
    _store(results, "score_time", score_times, n_splits, n_candidates)

    if not multimetric:
        _store(
            results,
            "test_score",
            test_scores,
            n_splits,
            n_candidates,
            splits=True,
            rank=True,
            weights=weights,
        )
        if train_scores is not None:
            _store(
                results,
                "train_score",
                train_scores,
                n_splits,
                n_candidates,
                splits=True,
            )
    else:
        for key in multimetric:
            _store(
                results,
                "test_{}".format(key),
                test_scores[key],
                n_splits,
                n_candidates,
                splits=True,
                rank=True,
                weights=weights,
            )
        if train_scores is not None:
            for key in multimetric:
                _store(
                    results,
                    "train_{}".format(key),
                    train_scores[key],
                    n_splits,
                    n_candidates,
                    splits=True,
                )

    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(
        lambda: MaskedArray(np.empty(n_candidates), mask=True, dtype=object)
    )
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)
    return results


def fit_best(estimator, params, X, y, fit_params):
    estimator = copy_estimator(estimator).set_params(**params)
    estimator.fit(X, y, **fit_params)
    return estimator
