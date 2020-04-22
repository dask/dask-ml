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
    def __init__(self, splits, pairwise=False, cache=True, num_train_samples=None ):
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
                self.num_train_samples
            ),
        )

    def num_test_samples(self):
        return np.array(
            [i.sum() if i.dtype == bool else len(i) for i in pluck(1, self.splits)]
        )

    def extract(self, X, y, n, is_x=True, is_train_folds=True):
        if is_x:
            if self.pairwise:
                return self._extract_pairwise(X, y, n, is_train_folds=is_train_folds)
            return self._extract(X, y, n, is_x=True, is_train_folds=is_train_folds)
        if y is None:
            return None
        return self._extract(X, y, n, is_x=False, is_train_folds=is_train_folds)

    def extract_param(self, key, x, n, is_train_folds=True):
        '''
            extract_param extracts the fit_params associated with a set of folds either train folds or test fold.
            Also supports caching similar to other extraction methods
        '''
        if self.cache is not None and (n, key, is_train_folds) in self.cache:
            return self.cache[n, key, is_train_folds]

        inds = self.splits[n][0] if is_train_folds else self.splits[n][1]
        out = _index_param_value( self.num_train_samples, x, inds)

        if self.cache is not None:
            self.cache[n, key, is_train_folds] = out
        return out

    def _extract(self, X, y, n, is_x=True, is_train_folds=True):
        if self.cache is not None and (n, is_x, is_train_folds) in self.cache:
            return self.cache[n, is_x, is_train_folds]

        inds = self.splits[n][0] if is_train_folds else self.splits[n][1]
        result = _safe_indexing(X if is_x else y, inds)

        if self.cache is not None:
            self.cache[n, is_x, is_train_folds] = result
        return result

    def _extract_pairwise(self, X, y, n, is_train_folds=True):
        if self.cache is not None and (n, True, is_train_folds) in self.cache:
            return self.cache[n, True, is_train_folds]

        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have "
                "to be passed as arrays or sparse matrices."
            )
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        train, test = self.splits[n]
        result = X[np.ix_(train if is_train_folds else test, train)]

        if self.cache is not None:
            self.cache[n, True, is_train_folds] = result
        return result


def cv_split(cv, X, y, groups, is_pairwise, cache):
    check_consistent_length(X, y, groups)
    return CVCache(list(cv.split(X, y, groups)), is_pairwise, cache, _num_samples(X))


def cv_n_samples(cvs):
    return cvs.num_test_samples()


def cv_extract(cvs, X, y, is_X, is_train_folds, n):
    return cvs.extract(X, y, n, is_X, is_train_folds)


def cv_extract_params(cvs, keys, vals, n, is_train_folds):
    '''
        cv_extract_params the fit parameters of the fold sets (train folds or test fold)
    '''
    return {k: cvs.extract_param(tok, v, n, is_train_folds) for (k, tok), v in zip(keys, vals)}

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

def _apply_scorer(estimator, X, y, scorer, sample_weight):
    """Applies the scorer to the estimator, given the data and sample_weight.

    If ``sample_weight`` is None ``sample_weight`` WILL
    NOT be passed to ``scorer``; otherwise, it will be passed.

    In the event that ``sample_weight`` is provided and used but ``scorer``
    doesn't accept a ``sample_weight`` parameter, then a ``TypeError`` should
    likely be raised.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object that was used to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like
        The target variable to try to predict in the case of
        supervised learning.  (May be None)

    scorer : A single callable.
        Should return a single float.

        The callable object / fn should have signature
        ``scorer(estimator, X, y, sample_weight=None)`` if ``sample_weight``.

    sample_weight : array-like, shape (y)
        sample weights to use during metric calculation.  May be None.

    Returns
    -------
    score : float
        Score returned by ``scorer`` applied to ``X`` and ``y`` given
        ``sample_weight``.
    """
    if sample_weight is None:
        if y is None:
            score = scorer(estimator, X)
        else:
            score = scorer(estimator, X, y)
    else:
        try:
            # Explicitly force the sample_weight parameter so that an error
            # will be raised in the event that the scorer doesn't take a
            # sample_weight argument.  This is preferable to passing it as
            # a keyword args dict in the case that it just ignores parameters
            # that are not accepted by the scorer.
            if y is None:
                score = scorer(estimator, X, sample_weight=sample_weight)
            else:
                score = scorer(estimator, X, y, sample_weight=sample_weight)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(
                    (
                        "Attempted to use 'sample_weight' for training "
                        "but supplied a scorer that doesn't accept a "
                        "'sample_weight' parameter."
                    ),
                    e,
                )
            else:
                raise e
    return score

def _score(est, X, y, scorer, sample_weight):
    if est is FIT_FAILURE:
        return FIT_FAILURE
    if isinstance(scorer, Mapping):
        return {
            k: _apply_scorer(est, X, y, v, sample_weight) for k, v in scorer.items()
        }
    return _apply_scorer(est, X, y, scorer, sample_weight)


def score(
     est_and_time,
     X_test,
     y_test,
     X_train,
     y_train,
     scorer,
     error_score,
     sample_weight=None,
     eval_sample_weight=None,
):
    est, fit_time = est_and_time
    start_time = default_timer()

    try:
        test_score = _score(est, X_test, y_test, scorer, eval_sample_weight)
    except Exception:
        if error_score == "raise":
            raise
        else:
            score_time = default_timer() - start_time
            return fit_time, error_score, score_time, error_score
    score_time = default_timer() - start_time
    if X_train is None:
        return fit_time, test_score, score_time
    train_score = _score(est, X_train, y_train, scorer, sample_weight)
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
    test_fit_params=None,
    return_train_score=True,
):
    X_train = cv.extract(X, y, n, True, True)
    y_train = cv.extract(X, y, n, False, True)
    X_test = cv.extract(X, y, n, True, False)
    y_test = cv.extract(X, y, n, False, False)

    '''
        Support for lightGBM evaluation data sets within folds.         
        https: // lightgbm.readthedocs.io / en / latest / pythonapi / lightgbm.LGBMClassifier.html
        
        Set the test set to the corresponding part of the eval set with the test folds index.
        Without this you can only use a set of corresponding size to train folds as eval_data_set requiring more data in the fit function. 
    '''
    if 'eval_set' in fit_params:
        fit_params['eval_set'] = test_fit_params['eval_set']
        fit_params['eval_names'] = test_fit_params['eval_names']
        fit_params['eval_sample_weight'] = test_fit_params['eval_sample_weight']

    est_and_time = fit(est, X_train, y_train, error_score, fields, params, fit_params)
    if not return_train_score:
        X_train = y_train = None

    eval_sample_weight_train, eval_sample_weight_test = None, None
    if fit_params is not None:
        '''
            NOTE: To be back-compatible with dask-ml defaults you could add a boolean (legacy_mode) that can skip the following block if (legacy_mode==True)
        '''
        eval_weight_source = "eval_sample_weight" if "eval_sample_weight" in fit_params else "sample_weight"
        if eval_weight_source in fit_params and fit_params[eval_weight_source] is not None:
            eval_sample_weight_train = fit_params[eval_weight_source]
        if eval_weight_source in test_fit_params and test_fit_params[eval_weight_source] is not None:
            eval_sample_weight_test = test_fit_params[eval_weight_source]

    return score(est_and_time, X_test, y_test, X_train, y_train, scorer, error_score, sample_weight=eval_sample_weight_train, eval_sample_weight=eval_sample_weight_test)


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