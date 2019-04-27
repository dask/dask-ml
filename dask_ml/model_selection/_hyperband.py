from __future__ import division

import logging
import math
from warnings import warn

import numpy as np
from sklearn.metrics.scorer import check_scoring
from sklearn.utils import check_random_state
from tornado import gen

from ._incremental import IncrementalSearchCV
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
    brackets : Dict[int, Tuple[int, int]]
        A dictionary of the form {bracket_id: (n_models, n_initial_iter)}

    Notes
    -----
    The bracket index is a measure of how strong that n,r combination
    adapts to prior input. i.e., a bracket ID of 0 means "doesn't adapt
    at all" and bracket index of 5 means "adapts pretty strongly"

    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [int(math.ceil(B / R * eta ** s / (s + 1))) for s in brackets]
    R = [int(R * eta ** -s) for s in brackets]
    return {b: (n, r) for b, n, r in zip(brackets, N, R)}


class HyperbandSearchCV(IncrementalSearchCV):
    """Find the best parameters for a particular model with an adaptive
    cross-validation algorithm.

    Hyperband will find close to the best possible parameters with
    the given computational budget [*]_ by spending more time training
    high-performing estimators [1]_. This means that Hyperband stops training
    estimators that perform poorly.

    Hyperband only requires computational budget as input, and
    does not require a trade-off between "evaluate many parameters
    for a short time" and "train a few parameters for a long time"
    like RandomizedSearchCV.

    .. [*] After :math:`N` ``partial_fit`` calls the estimator Hyperband
       produces will be close to the best possible estimator that :math:`N`
       ``partial_fit`` calls could ever produce with high probability (where
       "close" means "within log terms").

    Parameters
    ----------
    estimator : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of Scikit-Learn's
        BaseEstimator

    parameters : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    max_iter : int
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge. See the notes on how to set this parameter.

    aggressiveness : int, default=3
        How aggressive to be in culling off the different estimators. Higher
        values imply higher confidence in scoring (or that
        the hyperparameters influence the ``estimator.score`` more
        than the data). Theory suggests ``aggressiveness=3`` is close to
        optimal. ``aggressiveness=4`` has higher confidence that is likely
        suitable for initial exploration.

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

        If None, the estimator's default scorer (if available) is used.

    patience : int, default False
        Maximum number of non-improving scores before model training is
        stopepd. Off by default.

    tol : float, default 0.001
        The required level of improvement to consider stopping training on
        that model. The most recent score must be at at most ``tol`` better
        than the all of the previous ``patience`` scores for that model.
        Increasing ``tol`` will tend to reduce training time, at the cost
        of (potentially) worse estimators.

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

    Attributes
    ----------
    cv_results_ : Dict[str, np.ndarray]
        This dictionary has keys

        * ``mean_partial_fit_time``
        * ``mean_score_time``
        * ``std_partial_fit_time``
        * ``std_score_time``
        * ``test_score``
        * ``rank_test_score``
        * ``model_id``
        * ``partial_fit_calls``
        * ``params``
        * ``param_{key}``, where ``key`` is every key in ``params``.

        The values in the ``test_score`` key correspond to the last score a model
        received on the hold out dataset. The key ``model_id`` corresponds with
        ``history_``. This dictionary can be imported into Pandas.

        In the ``model_id``, the bracket ID prefix corresponds to how strongly
        that bracket adapts to history. i.e., ``bracket=0`` corresponds to a
        completely passive bracket that doesn't adapt at all.

    model_history_ : dict of lists of dict
        A dictionary of each models history. This is a reorganization of
        ``history_``: the same information is present but organized per model.

        This data has the structure  ``{model_id: hist}`` where ``hist`` is a
        subset of ``history_`` and ``model_id`` are model identifiers.

    history_ : list of dicts
        Information about each model after each ``partial_fit`` call. Each dict
        the keys

        * ``partial_fit_time``
        * ``score_time``
        * ``score``
        * ``model_id``
        * ``params``
        * ``partial_fit_calls``

        The key ``model_id`` corresponds to the ``model_id`` in ``cv_results_``.
        This list of dicts can be imported into Pandas.

    best_estimator_ : BaseEstimator
        The model with the highest validation score among all the models
        retained by the "inverse decay" algorithm.

    best_score_ : float
        Score achieved by ``best_estimator_`` on the vaidation set after the
        final call to ``partial_fit``.

    best_index_ : int
        Index indicating which estimator in ``cv_results_`` corresponds to
        the highest score.

    best_params_ : dict
        Dictionary of best parameters found on the hold-out data.

    scorer_ :
        The function used to score models, which has a call signature of
        ``scorer_(estimator, X, y)``.

    n_splits_ : int
        Number of cross validation splits.

    multimetric_ : bool
        Whether this cross validation search uses multiple metrics.

    Notes
    -----
    To set ``max_iter`` and the chunk size for ``X`` and ``y``, it is required
    to know

    * how many "epochs" or "passes through the data ``X``" to train the model
      for (``epochs`` below)
    * a rough idea of how many hyper-parameter combinations to sample
      (``params_to_sample`` below)

    To determine the chunk size and ``max_iter``,

    1. Let the chunks size be ``chunk_size = epochs * len(X) / params_to_sample``
    2. Let ``max_iter = params_to_sample``

    where ``len(X)`` is the number of examples. Then, every estimator sees no
    more than ``max_iter * chunk_size = len(X) * epochs`` examples.
    Hyperband will actually sample some more hyper-parameter combinations,
    so a rough idea of parameters to sample works.

    For instance, let's say about 200 or 300 hyper-parameters need to be tested
    to effectively search the possible hyper-parameters and estimators need a
    little less than 100 epochs to converge. This will mean the dataset should
    have chunks that are about 1/3rd of the entire dataset, and ``max_iter`` is
    between 200 and 300.

    If the search space is larger or more complex, choose to evaluate more
    hyper-parameters. Increase the parameters to sample by a factor of 2 by:

    * increasing ``params_to_sample`` by a factor of 2
    * decreasing ``chunk_size`` by a factor of 2.

    There are some limitations to the `current` implementation of Hyperband:

    1. The full dataset is requested to be in distributed memory
    2. The testing dataset should fit in the memory of a single worker

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560

    """

    def __init__(
        self,
        estimator,
        parameters,
        max_iter=81,
        aggressiveness=3,
        test_size=None,
        patience=False,
        tol=1e-3,
        random_state=None,
        scoring=None,
    ):
        self.max_iter = max_iter
        self.aggressiveness = aggressiveness

        super(HyperbandSearchCV, self).__init__(
            estimator,
            parameters,
            max_iter=max_iter,
            patience=patience,
            tol=tol,
            test_size=test_size,
            random_state=random_state,
            scoring=scoring,
        )

    def _get_SHAs(self, brackets):

        patience = _get_patience(self.patience, self.max_iter, self.aggressiveness)

        # This is the first time self.random_state is used after
        # HyperbandSearchCV.fit is called.
        seed_start = check_random_state(self.random_state).randint(2 ** 31)
        self._SHA_seed = seed_start

        # These brackets are ordered by adaptivity; bracket=0 is least adaptive
        SHAs = {
            b: SuccessiveHalvingSearchCV(
                self.estimator,
                self.parameters,
                n_initial_parameters=n,
                n_initial_iter=r,
                aggressiveness=self.aggressiveness,
                max_iter=self.max_iter,
                patience=patience,
                tol=self.tol,
                test_size=self.test_size,
                random_state=seed_start + b if b != 0 else self.random_state,
                scoring=self.scoring,
            )
            for b, (n, r) in brackets.items()
        }
        return SHAs

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        X = self._check_array(X)
        y = self._check_array(y, ensure_2d=False)
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        if self.max_iter < 1:
            raise ValueError("max_iter < 1 is not supported")

        brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)

        SHAs = self._get_SHAs(brackets)
        # Which bracket to run first? Going to go with most adaptive;
        # hopefully less adaptive can fill in for any blank spots
        #
        # _brackets_ids is ordered from largest to smallest
        _brackets_ids = list(reversed(sorted(SHAs)))
        _SHAs = yield [SHAs[b]._fit(X, y, **fit_params) for b in _brackets_ids]
        SHAs = {b: SHA for b, SHA in zip(_brackets_ids, _SHAs)}

        # This for-loop rename estimator IDs and pulls out wall times
        key = "bracket={}-{}".format
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

        keys = {k for SHA in SHAs.values() for k in SHA.cv_results_.keys()}
        cv_results = {
            k: sum([SHA.cv_results_[k].tolist() for SHA in SHAs.values()], [])
            for k in keys
        }
        cv_results = {k: np.array(v) for k, v in cv_results.items()}

        scores = {b: SHA.best_score_ for b, SHA in SHAs.items()}
        best_bracket = max(scores, key=scores.get)

        best_estimator = SHAs[best_bracket].best_estimator_

        estimator_history = {
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
            {b: SHA.history_ for b, SHA in SHAs.items()}, brackets.keys(), SHAs, key
        )

        self.metadata_ = {
            "estimators": sum(m["estimators"] for m in meta.values()),
            "partial_fit_calls": sum(m["partial_fit_calls"] for m in meta.values()),
            "brackets": meta,
        }

        self.best_index_ = int(best_index)
        self.best_estimator_ = best_estimator
        self.best_score_ = scores[best_bracket]
        self.best_params_ = cv_results["params"][best_index]
        self.scorer_ = scorer

        self.model_history_ = estimator_history
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
            * ``estimators``, the total number of estimators created.
            * ``brackets``, each of which has the same two keys as above.

        Notes
        ------
        This algorithm runs several loops in an "embarassingly parallel"
        manner. The ``brackets`` key represents each of these loops.

        """
        bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.aggressiveness)
        num_models = sum(b["estimators"] for b in bracket_info)
        for bracket in bracket_info:
            bracket["decisions"].update({1})
            bracket["decisions"] = sorted(list(bracket["decisions"]))
        num_partial_fit = sum(b["partial_fit_calls"] for b in bracket_info)
        bracket_info = list(reversed(sorted(bracket_info, key=lambda x: x["bracket"])))

        brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)
        SHAs = self._get_SHAs(brackets)
        for bracket in bracket_info:
            b = bracket["bracket"]
            bracket["SuccessiveHalvingSearchCV params"] = _get_SHA_params(SHAs[b])

        info = {
            "partial_fit_calls": num_partial_fit,
            "estimators": num_models,
            "brackets": {"bracket=" + str(b["bracket"]): b for b in bracket_info},
        }
        return info


def _get_meta(hists, brackets, SHAs, key):
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
        decisions = {hi["partial_fit_calls"] for h in hist.values() for hi in h}
        meta_["bracket=" + str(bracket)] = {
            "decisions": sorted(list(decisions)),
            "estimators": len(hist),
            "bracket": bracket,
            "partial_fit_calls": sum(calls.values()),
            "SuccessiveHalvingSearchCV params": _get_SHA_params(SHAs[bracket]),
        }
    return meta_, history_


def _get_SHA_params(SHA):
    return {
        k: v
        for k, v in SHA.get_params().items()
        if "estimator_" not in k  # and k != "param_distribution"
    }


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
        hist = {
            "num_estimators": n,
            "estimators": {n: 0 for n in range(n)},
            "decisions": [],
        }
        for i in range(s + 1):
            n_i = math.floor(n * eta ** -i)
            r_i = np.round(r * eta ** i).astype(int)
            L = {model: r_i for model in T}
            hist["estimators"].update(L)
            hist["decisions"] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T) if i < to_keep}
        hists[s] = hist
    info = [
        {
            "bracket": k,
            "estimators": hist["num_estimators"],
            "partial_fit_calls": sum(hist["estimators"].values()),
            "decisions": {int(h) for h in hist["decisions"]},
        }
        for k, hist in hists.items()
    ]
    return info


def _get_patience(patience, max_iter, aggressiveness):
    if not isinstance(patience, bool) and patience < max(max_iter // aggressiveness, 1):
        msg = (
            "Careful. patience={}, but values of patience=True (or "
            "patience>={}) are recommended.\n\n"
            "The goal of `patience` is to stop training estimators that have "
            "already converged *when few estimators remain*."
            "Setting patience=True accomplishes this goal.\n\n Please "
            "continue with caution or good reason"
        )
        warn(msg.format(patience, max_iter // aggressiveness))
    elif isinstance(patience, bool) and patience:
        return max(max_iter // aggressiveness, 1)
    elif isinstance(patience, bool) and not patience:
        return False
    return int(patience)
