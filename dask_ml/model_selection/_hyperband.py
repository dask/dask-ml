from __future__ import division

import logging
import math
from time import time
from warnings import warn

import dask.array as da
import numpy as np
from dask.distributed import default_client
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import ParameterSampler
from sklearn.utils.metaestimators import if_delegate_has_method
from tornado import gen

from ._incremental import INC_ATTRS, IncrementalSearchCV, fit as _incremental_fit
from ._search import DaskBaseSearchCV
from ._split import train_test_split
from ._successive_halving import SuccessiveHalvingSearchCV

logger = logging.getLogger(__name__)


def _get_hyperband_params(R, eta=3):
    """
    Parameters
    ----------
    R : int
        The maximum number of iterations desired.

    Returns
    -------
    N : list
        The number of models for each bracket
    R : list
        The number of iterations for each bracket
    brackets : list
        The bracket identifier.

    Notes
    -----
    The bracket index is a measure of how strong that n,r combination
    adapts to prior input. i.e., a bracket ID of 0 means "doesn't adapt
    at all" and bracket index of 5 means "adapts pretty strongly"

    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta ** s / (s + 1)) for s in brackets]
    R = [int(R * eta ** -s) for s in brackets]
    return list(map(int, N)), R, brackets


DOC = (
    """Find the best parameters for a particular model with an adaptive
    cross-validation algorithm.

    Hyperband will find close to the best possible
    parameters with the given computational budget [1]_.* It does this by
    focusing on spending time training high-performing models. This means that
    it stops training models that perform poorly.

    This algorithm performs well, has theoritical justification [1]_ and only
    requires computational budget as input. It does not require a trade-off
    between "evaluate many parameters" and "train for a long time" like
    RandomizedSearchCV.

    :sup:`* This will happen with high probability, and "close" means "within
    a log factor of the lower bound on the best possible score"`

    Parameters
    ----------
    model : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of Scikit-Learn's
        BaseEstimator

    params : dict, list
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    max_iter : int
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge. See the notes on how to set this parameter.

    aggressiveness : int, default=3
        How aggressive to be in model tuning. It is not recommended to change
        this value, and if changed we recommend ``eta=4``.
        Some theory behind Hyperband suggests ``eta=np.e``. Higher
        values imply higher confidence in model selection.

    test_size : float
        Fraction of the dataset to hold out for computing test scores.
        Defaults to the size of a single partition of the input training set

        .. note::

           The testing dataset should fit in memory on a single machine.
           Adjust the ``test_size`` parameter as necessary to achieve this.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    patience : int, default False
        Maximum number of non-improving scores before we stop training a
        model. Off by default.

    tol : float, default 0.001
        The required level of improvement to consider stopping training on
        that model. The most recent score must be at at most ``tol`` better
        than the all of the previous ``patience`` scores for that model.
        Increasing ``tol`` will tend to reduce training time, at the cost
        of worse models.

    **kwargs : dict, optional
        Parameters to pass to
        :func:`~dask_ml.model_selection.IncrementalSearchCV`

    Examples
    --------
    >>> import numpy as np
    >>> from dask_ml.model_selection import HyperbandSearchCV
    >>> from dask_ml.datasets import make_classification
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, y = make_classification(chunks=20)
    >>> est = SGDClassifier(tol=1e-3)
    >>> param_dist = {'alpha': np.logspace(-4, 0, num=1000),
    >>>               'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    >>>               'average': [True, False]}
    >>>
    >>> search = HyperbandSearchCV(est, param_dist)
    >>> search.fit(X, y, classes=np.unique(y))
    >>> search.best_params_
    {'loss': 'log', 'average': False, 'alpha': 0.0080502}

    """
    + INC_ATTRS
    + """
    Notes
    -----
    In ``model_id``, the bracket ID prefix corresponds to how strongly that
    bracket adapts to history. i.e., ``bracket=0`` corresponds to a completely
    passive bracket that doesn't adapt at all.

    Setting Hyperband parameters
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    To set ``max_iter`` and the chunk size for ``X`` and ``y``, it is required
    to know

    * how many "epochs" or "passes through ``X``" to train the model for
      (``epochs`` below)
    * a rough idea of how many hyper-parameter combinations to sample (``params_to_sample`` below)

    To determine the chunk size and ``max_iter``,

    1. Let ``max_iter = params_to_sample``
    2. Let the chunks size be ``chunks_size = epochs * len(X) / params_to_sample``

    Then, the estimator that sees the most examples see
    ``max_iter * chunks_size = len(X) * epochs`` examples. Hyperband will
    actually sample some more hyper-parameter combinations, so a rough idea of
    parameters to sample works.

    If the search space is complex, evaluate more estimators initially. Increase
    ``params_to_sample`` by a factor of 2, and decrease ``chunk_size`` by a
    factor of 2.

    Limitations
    ^^^^^^^^^^^
    There are some limitations to the `current` implementation of Hyperband:

    1. The full dataset is requested to be in distributed memory
    2. The testing dataset should fit in the memory of a single worker

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560

    """
)


class HyperbandSearchCV(IncrementalSearchCV):
    __doc__ = DOC

    def __init__(
        self,
        estimator,
        param_distribution,
        max_iter,
        aggressiveness=3,
        test_size=None,
        patience=False,
        tol=1e-3,
        scores_per_fit=1,
        random_state=None,
        scoring=None,
    ):
        self.aggressiveness = aggressiveness
        self.param_distribution = param_distribution

        super(HyperbandSearchCV, self).__init__(
            estimator,
            param_distribution,
            max_iter=max_iter,
            patience=patience,
            tol=tol,
            test_size=test_size,
            scores_per_fit=scores_per_fit,
            random_state=random_state,
            scoring=scoring,
        )

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        X, y = self._check_array(X, y)
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        if not isinstance(self.patience, bool) and self.patience < max(
            self.max_iter // self.aggressiveness, 1
        ):
            msg = (
                "Careful. patience={}, but values of patience=True (or maybe "
                "patience>={}) are recommended.\n\n"
                "The goal of `patience` is to stop training estimators that have "
                "already converged *when few estimators remain*."
                "Setting patience=True accomplishes this goal. Please continue "
                "with caution or good reason"
            )
            warn(msg.format(self.patience, self.max_iter // self.aggressiveness))
        if self.patience:
            patience = max(self.max_iter // self.aggressiveness, 1)
        else:
            patience = False

        N, R, brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)

        # These brackets are ordered by adaptivity; most adaptive comes
        # first
        SHAs = [
            (
                b,
                SuccessiveHalvingSearchCV(
                    self.estimator,
                    self.param_distribution,
                    n,
                    r,
                    limit=b + 1,
                    aggressiveness=self.aggressiveness,
                    patience=patience,
                    tol=self.tol,
                ),
            )
            for n, r, b in zip(N, R, brackets)
        ]
        # Which bracket to run first? Going to go with most adaptive;
        # hopefully less adaptive can fill in for any blank spots
        SHAs = yield {b: SHA.fit(X, y, **fit_params) for b, SHA in SHAs}

        # This for-loop rename estimator IDs and pulls out wall times
        key = lambda b, old: "bracket={}-{}".format(b, old)
        for b, SHA in SHAs.items():
            new_ids = {old: key(b, old) for old in SHA.cv_results_["model_id"]}
            SHA.cv_results_["model_id"] = np.array(
                [new_ids[old] for old in SHA.cv_results_["model_id"]]
            )
            SHA.model_history_ = {
                new_ids[old]: v for old, v in SHA.model_history_.items()
            }
            for hist in SHA.model_history_.values():
                for h in hist:
                    h["model_id"] = new_ids[h["model_id"]]
                    h["bracket"] = b

        keys = list(SHA.cv_results_.keys())
        cv_results = {
            k: sum([SHA.cv_results_[k].tolist() for SHA in SHAs.values()], [])
            for k in keys
        }
        cv_results = {k: np.array(v) for k, v in cv_results.items()}

        scores = {b: SHA.best_score_ for b, SHA in SHAs.items()}
        best_bracket = max(scores, key=scores.get)

        best_estimator = SHAs[best_bracket].best_estimator_

        model_history = {
            ident: hist
            for SHA in SHAs.values()
            for ident, hist in SHA.model_history_.items()
        }

        # Order history by time
        history = sum([SHA.history_ for b, SHA in SHAs.items()], [])
        idx = np.argsort([v["elapsed_wall_time"] for v in history])
        history = [history[i] for i in idx]

        best_model_id = SHAs[best_bracket].cv_results_["model_id"][
            SHAs[best_bracket].best_index_
        ]
        best_index = np.argwhere(np.array(cv_results["model_id"]) == best_model_id)
        best_index = best_index.flat[0]

        meta, _ = _get_meta(
            {b: SHA.history_ for b, SHA in SHAs.items()}, brackets, SHAs, key=key
        )

        self.metadata_ = {
            "models": sum(m["models"] for m in meta.values()),
            "partial_fit_calls": sum(m["partial_fit_calls"] for m in meta.values()),
            "brackets": meta,
        }

        self.best_index_ = int(best_index)
        self.best_estimator_ = best_estimator
        self.best_score_ = scores[best_bracket]
        self.best_params_ = cv_results["params"][best_index]
        self.scorer_ = scorer

        self.model_history_ = model_history
        self.history_ = history
        self.cv_results_ = cv_results

        self.multimetric_ = SHAs[best_bracket].multimetric_
        raise gen.Return(self)

    def metadata(self):
        """Get information about how much computation is required for
        :func:`~dask_ml.model_selection.HyperbandSearchCV.fit`. This can be called
        before or after  :func:`~dask_ml.model_selection.HyperbandSearchCV.fit`.

        Returns
        -------
        metadata : dict
            Information about the computation performed by ``fit``. Has keys

            * ``partial_fit_calls``, the total number of partial fit calls.
            * ``models``, the total number of models created.
            * ``brackets``, each of which has the same two keys as above.

        Notes
        ------
        This algorithm runs several loops in an "embarassingly parallel"
        manner. The ``brackets`` key represents each of these loops.

        """
        bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.aggressiveness)
        num_models = sum(b["models"] for b in bracket_info)
        for bracket in bracket_info:
            bracket["iters"].update({1})
            bracket["iters"] = sorted(list(bracket["iters"]))
        num_partial_fit = sum(b["partial_fit_calls"] for b in bracket_info)
        bracket_info = list(reversed(sorted(bracket_info, key=lambda x: x["bracket"])))

        N, R, brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)
        SHAs = {
            b: SuccessiveHalvingSearchCV(
                self.estimator,
                self.param_distribution,
                n,
                r,
                limit=b + 1,
                aggressiveness=self.aggressiveness,
            )
            for n, r, b in zip(N, R, brackets)
        }
        for bracket in bracket_info:
            b = bracket["bracket"]
            bracket["SuccessiveHalvingSearchCV params"] = _get_SHA_params(SHAs[b])

        info = {
            "partial_fit_calls": num_partial_fit,
            "models": num_models,
            "brackets": {"bracket=" + str(b.pop("bracket")): b for b in bracket_info},
        }
        return info


def _get_meta(hists, brackets, SHAs, key=None):
    if key is None:
        key = lambda bracket, ident: "bracket={}-{}".format(bracket, ident)
    meta_ = {}
    history_ = {}
    for bracket in brackets:
        hist = hists[bracket]

        info_hist = {key(bracket, h["model_id"]): [] for h in hist}
        for h in hist:
            info_hist[key(bracket, h["model_id"])] += [h]
        hist = info_hist
        history_.update(hist)

        calls = {k: max(hi["partial_fit_calls"] for hi in h) for k, h in hist.items()}
        iters = {hi["partial_fit_calls"] for h in hist.values() for hi in h}
        meta_["bracket=" + str(bracket)] = {
            "iters": sorted(list(iters)),
            "models": len(hist),
            "partial_fit_calls": sum(calls.values()),
            "SuccessiveHalvingSearchCV params": _get_SHA_params(SHAs[bracket]),
        }
    return meta_, history_


def _get_SHA_params(SHA):
    return {
        k: v
        for k, v in SHA.get_params().items()
        if "estimator" not in k and k != "param_distribution"
    }


def _get_cv_results(hists, params, key=None):
    if key is None:
        key = lambda bracket, ident: "bracket={}-{}".format(bracket, ident)
    info = {
        key(bracket, h["model_id"]): {
            "bracket": bracket,
            "score": None,
            "partial_fit_calls": -np.inf,
            "fit_times": [],
            "score_times": [],
            "model_id": h["model_id"],
            "params": param,
        }
        for bracket, hist in hists.items()
        for h, param in zip(hist, params[bracket])
    }

    for bracket, hist in hists.items():
        for h in hist:
            k = key(bracket, h["model_id"])
            if info[k]["partial_fit_calls"] < h["partial_fit_calls"]:
                info[k]["partial_fit_calls"] = h["partial_fit_calls"]
                info[k]["score"] = h["score"]
                info[k]["fit_times"] += [h["partial_fit_time"]]
                info[k]["score_times"] += [h["score_time"]]

    info = list(info.values())
    scores = np.array([v["score"] for v in info])
    idents = [key(v["bracket"], v["model_id"]) for v in info]

    best_idx = int(np.argmax(scores))
    best_ident = idents[best_idx]

    ranks = np.argsort(-1 * scores) + 1

    def get_map(fn, key, list_):
        return np.array([fn(dict_[key]) for dict_ in list_])

    cv_results = {
        "params": [v["params"] for v in info],
        "test_score": scores,
        "mean_test_score": scores,  # for sklearn comptability
        "rank_test_score": ranks,
        "mean_partial_fit_time": get_map(np.mean, "fit_times", info),
        "std_partial_fit_time": get_map(np.std, "fit_times", info),
        "mean_score_time": get_map(np.mean, "score_times", info),
        "std_score_time": get_map(np.std, "score_times", info),
        "partial_fit_calls": [v["partial_fit_calls"] for v in info],
        "model_id": idents,
    }
    params = sum(params.values(), [])
    all_params = {k for param in params for k in param}
    flat_params = {
        "param_" + k: [param.get(k, None) for param in params] for k in all_params
    }
    cv_results.update(flat_params)
    return cv_results, best_idx, best_ident


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper [1]_.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh,
       and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(int(s_max + 1)))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta ** s / (s + 1)))
        r = R * eta ** -s
        r = int(r)
        T = set(range(n))
        hist = {"num_models": n, "models": {n: 0 for n in range(n)}, "iters": []}
        for i in range(s + 1):
            n_i = math.floor(n * eta ** -i)
            r_i = np.round(r * eta ** i).astype(int)
            L = {model: r_i for model in T}
            hist["models"].update(L)
            hist["iters"] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T) if i < to_keep}
        hists[s] = hist
    info = [
        {
            "bracket": k,
            "models": hist["num_models"],
            "partial_fit_calls": sum(hist["models"].values()),
            "iters": {int(h) for h in hist["iters"]},
        }
        for k, hist in hists.items()
    ]
    return info
