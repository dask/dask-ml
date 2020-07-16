from __future__ import division

import asyncio
import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from warnings import warn

import numpy as np
from sklearn.utils import check_random_state

from ._incremental import BaseIncrementalSearchCV
from ._successive_halving import SuccessiveHalvingSearchCV

logger = logging.getLogger(__name__)

BracketInfo = Dict[str, Any]


def _get_hyperband_params(R, eta=3):
    """
    Parameters
    ----------
    R : int
        The maximum number of iterations desired.
    eta : int
        How aggressive to be in the search

    Returns
    -------
    brackets : Dict[int, Tuple[int, int]]
        A dictionary of the form {bracket_id: (n_models, n_initial_iter)}

    Notes
    -----
    The bracket index is a measure of how strong that n,r combination
    adapts to prior input. i.e., a bracket ID of 0 means "doesn't adapt
    at all" and bracket index of 5 means "adapts pretty strongly"

    ``R`` and ``eta`` are the terminology that the Hyperband paper uses [1]_.

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar. https://arxiv.org/abs/1603.06560

    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [int(math.ceil(B / R * eta ** s / (s + 1))) for s in brackets]
    R = [int(R * eta ** -s) for s in brackets]
    return {b: (n, r) for b, n, r in zip(brackets, N, R)}


class HyperbandSearchCV(BaseIncrementalSearchCV):
    """Find the best parameters for a particular model with an adaptive
    cross-validation algorithm.

    Hyperband will find close to the best possible parameters with
    the given computational budget [*]_ by spending more time training
    high-performing estimators [1]_. This means that Hyperband stops training
    estimators that perform poorly -- at it's core, Hyperband is an early
    stopping scheme for RandomizedSearchCV.

    Hyperband does not require a trade-off between "evaluate many parameters
    for a short time" and "train a few parameters for a long time"
    like RandomizedSearchCV.

    Hyperband requires one input which requires knowing how long
    to train the best performing estimator via ``max_iter``.
    The other implicit input (the Dask array chuck size) requires
    a rough estimate of how many parameters to sample. Specification details
    are in :ref:`Notes <hyperband-notes>`.

    .. [*] After :math:`N` ``partial_fit`` calls the estimator Hyperband
       produces will be close to the best possible estimator that :math:`N`
       ``partial_fit`` calls could ever produce with high probability (where
       "close" means "within log terms of the expected best possible score").

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each hyperparameter
        combination. This is assumed to implement the scikit-learn estimator
        interface. Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed. The estimator must implement
        ``partial_fit``, ``set_params``, and work well with ``clone``.

    parameters : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    max_iter : int
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge. See :ref:`Notes <hyperband-notes>` for details on
        setting this parameter.

    aggressiveness : int, default=3
        How aggressive to be in culling off the different estimators. Higher
        values imply higher confidence in scoring (or that
        the hyperparameters influence the ``estimator.score`` more
        than the data). Theory suggests ``aggressiveness=3`` is close to
        optimal. ``aggressiveness=4`` has higher confidence that is likely
        suitable for initial exploration.

    patience : int, default False
        If specified, training stops when the score does not increase by
        ``tol`` after ``patience`` calls to ``partial_fit``. Off by default.
        A ``patience`` value is automatically selected if ``patience=True`` to
        work well with the Hyperband model selection algorithm.

    tol : float, default 0.001
        The required level of improvement to consider stopping training on
        that model when ``patience`` is specified.  Increasing ``tol`` will
        tend to reduce training time at the cost of (potentially) worse
        estimators.

    test_size : float
        Fraction of the dataset to hold out for computing test/validation
        scores. Defaults to the size of a single partition of
        the input training set.

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

    verbose : bool, float, int, optional, default: False
        If False (default), don't print logs (or pipe them to stdout). However,
        standard logging will still be used.

        If True, print logs and use standard logging.

        If float, print/log approximately ``verbose`` fraction of the time.

    prefix : str, optional, default=""
        While logging, add ``prefix`` to each message.

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
    metadata and metadata_ : dict[str, Union(int, dict)]

        These dictionaries describe the computation performed, either
        before computation happens with ``metadata`` or after computation
        happens with ``metadata_``. These dictionaries both have keys

        * ``n_models``, an int representing how many models will be/is created.
        * ``partial_fit_calls``, an int representing how many times
           ``partial_fit`` will be/is called.
        * ``brackets``, a list of the brackets that Hyperband runs. Each
          bracket has different values for training time importance and
          hyperparameter importance. In addition to ``n_models`` and
          ``partial_fit_calls``, each element in this list has keys
            * ``bracket``, an int the bracket ID. Each bracket corresponds to
              a different levels of training time importance.
              For bracket 0, training time is important. For the highest
              bracket, training time is not important and models are killed
              aggressively.
            * ``SuccessiveHalvingSearchCV params``, a dictionary used to create
              the different brackets. It does not include the
              ``estimator`` or ``parameters`` parameters.
            * ``decisions``, the number of ``partial_fit`` calls Hyperband makes
              before making decisions.

        These dictionaries are the same if ``patience`` is not specified. If
        ``patience`` is specified, it's possible that less training is
        performed, and ``metadata_`` will reflect that (though ``metadata``
        won't).

    cv_results_ : Dict[str, np.ndarray]
        A dictionary that describes how well each model has performed.
        It contains information about every model regardless if it reached
        ``max_iter``.  It has keys

        * ``mean_partial_fit_time``
        * ``mean_score_time``
        * ``std_partial_fit_time``
        * ``std_score_time``
        * ``test_score``
        * ``rank_test_score``
        * ``model_id``
        * ``partial_fit_calls``
        * ``params``
        * ``param_{key}``, where ``{key}`` is every key in ``params``.
        * ``bracket``

        The values in the ``test_score`` key correspond to the last score a model
        received on the hold out dataset. The key ``model_id`` corresponds with
        ``history_``. This dictionary can be imported into a Pandas DataFrame.

        In the ``model_id``, the bracket ID prefix corresponds to the bracket
        in ``metadata``. Bracket 0 doesn't adapt to previous training at all;
        higher values correspond to more adaptation.

    history_ : list of dicts
        Information about each model after each ``partial_fit`` call. Each dict
        the keys

        * ``partial_fit_time``
        * ``score_time``
        * ``score``
        * ``model_id``
        * ``params``
        * ``partial_fit_calls``
        * ``elapsed_wall_time``

        The key ``model_id`` corresponds to the ``model_id`` in ``cv_results_``.
        This list of dicts can be imported into Pandas.

    model_history_ : dict of lists of dict
        A dictionary of each models history. This is a reorganization of
        ``history_``: the same information is present but organized per model.

        This data has the structure  ``{model_id: [h1, h2, h3, ...]}`` where
        ``h1``, ``h2`` and ``h3`` are elements of ``history_``
        and ``model_id`` is the model ID as in ``cv_results_``.

    best_estimator_ : BaseEstimator
        The model with the highest validation score as selected by
        the Hyperband model selection algorithm.

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

    Notes
    -----

    .. _hyperband-notes:

    To set ``max_iter`` and the chunk size for ``X`` and ``y``, it is required
    to estimate

    * the number of examples at least one model will see
      (``n_examples``). If 10 passes through the data are needed for
      the longest trained model, ``n_examples = 10 * len(X)``.
    * how many hyper-parameter combinations to sample (``n_params``)

    These can be rough guesses. To determine the chunk size and ``max_iter``,

    1. Let the chunks size be ``chunk_size = n_examples / n_params``
    2. Let ``max_iter = n_params``

    Then, every estimator sees no
    more than ``max_iter * chunk_size = n_examples`` examples.
    Hyperband will actually sample some more hyper-parameter combinations than
    ``n_examples`` (which is why rough guesses are adequate). For example,
    let's say

    * about 200 or 300 hyper-parameters need to be tested to effectively
      search the possible hyper-parameters
    * models need more than ``50 * len(X)`` examples but less than
      ``100 * len(X)`` examples.

    Let's decide to provide ``81 * len(X)`` examples and to sample 243
    parameters. Then each chunk will be 1/3rd the dataset and ``max_iter=243``.

    If you use ``HyperbandSearchCV``, please use the citation for [2]_

    .. code-block:: tex

        @InProceedings{sievert2019better,
            author    = {Scott Sievert and Tom Augspurger and Matthew Rocklin},
            title     = {{B}etter and faster hyperparameter optimization with {D}ask},
            booktitle = {{P}roceedings of the 18th {P}ython in {S}cience {C}onference},
            pages     = {118 - 125},
            year      = {2019},
            editor    = {Chris Calloway and David Lippa and Dillon Niederhut and David Shupe},  # noqa
            doi       = {10.25080/Majora-7ddc1dd1-011}
          }

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    .. [2] "Better and faster hyperparameter optimization with Dask", 2018 by
           S. Sievert, T. Augspurger, M. Rocklin.
           https://doi.org/10.25080/Majora-7ddc1dd1-011

    """

    def __init__(
        self,
        estimator,
        parameters,
        max_iter=81,
        aggressiveness=3,
        patience=False,
        tol=1e-3,
        test_size=None,
        random_state=None,
        scoring=None,
        verbose=False,
        prefix="",
    ):
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
            verbose=verbose,
            prefix=prefix,
        )

    def _get_SHAs(self, brackets):
        patience = _get_patience(
            self.patience, self.max_iter, self.aggressiveness, self.tol
        )

        # This is the first time self.random_state is used after
        # HyperbandSearchCV.fit is called.
        seed_start = check_random_state(self.random_state).randint(2 ** 31)
        self._SHA_seed = seed_start

        # These brackets are ordered by adaptivity; bracket=0 is least adaptive
        SHAs = {}
        for b, (n, r) in brackets.items():
            sha = SuccessiveHalvingSearchCV(
                self.estimator,
                self.parameters,
                n_initial_parameters=n,
                aggressiveness=self.aggressiveness,
                max_iter=self.max_iter,
                n_initial_iter=r,
                patience=patience,
                tol=self.tol,
                test_size=self.test_size,
                random_state=seed_start + b if b != 0 else self.random_state,
                scoring=self.scoring,
                verbose=self.verbose,
                prefix=f"{self.prefix}, bracket={b}",
            )
            SHAs[b] = sha
        return SHAs

    async def _fit(self, X, y, **fit_params):
        X, y, scorer = await self._validate_parameters(X, y)

        brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)
        SHAs = self._get_SHAs(brackets)

        # Which bracket to run first? Going to go with most adaptive;
        # that works best on one machine.
        # (though it doesn't matter a ton; _fit prioritizes high scores
        _brackets_ids = list(reversed(sorted(SHAs)))

        _SHAs = await asyncio.gather(
            *[SHAs[b]._fit(X, y, **fit_params) for b in _brackets_ids]
        )
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

        for b, SHA in SHAs.items():
            n = len(SHA.cv_results_["model_id"])
            SHA.cv_results_["bracket"] = np.ones(n, dtype=int) * b

        cv_keys = {k for SHA in SHAs.values() for k in SHA.cv_results_.keys()}

        cv_results = {
            k: [v for b in _brackets_ids for v in SHAs[b].cv_results_[k]]
            for k in cv_keys
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
            "n_models": sum(m["n_models"] for m in meta),
            "partial_fit_calls": sum(m["partial_fit_calls"] for m in meta),
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
        self._SuccessiveHalvings_ = SHAs
        return self

    @property
    def metadata(self) -> Dict[str, Union[int, List[BracketInfo]]]:
        bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.aggressiveness)
        num_models = sum(b["n_models"] for b in bracket_info)
        for bracket in bracket_info:
            bracket["decisions"] = sorted(list(bracket["decisions"]))
        num_partial_fit = sum(b["partial_fit_calls"] for b in bracket_info)
        bracket_info = list(reversed(sorted(bracket_info, key=lambda x: x["bracket"])))

        brackets = _get_hyperband_params(self.max_iter, eta=self.aggressiveness)
        SHAs = self._get_SHAs(brackets)
        for bracket in bracket_info:
            b = bracket["bracket"]
            bracket["SuccessiveHalvingSearchCV params"] = _get_SHA_params(SHAs[b])

        bracket_info = sorted(bracket_info, key=lambda x: x["bracket"])
        info = {
            "partial_fit_calls": num_partial_fit,
            "n_models": num_models,
            "brackets": bracket_info,
        }
        return info


def _get_meta(
    hists: Dict[int, List[Dict[str, Any]]],
    brackets: Iterable[int],
    SHAs: Dict[int, SuccessiveHalvingSearchCV],
    key: Callable[[int, int], str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:

    meta_ = []
    history_ = {}
    for bracket in brackets:
        _hist = hists[bracket]

        info_hist = {key(bracket, h["model_id"]): [] for h in _hist}
        for h in _hist:
            info_hist[key(bracket, h["model_id"])] += [h]
        hist = info_hist
        history_.update(hist)

        calls = {k: max(hi["partial_fit_calls"] for hi in h) for k, h in hist.items()}
        decisions = {hi["partial_fit_calls"] for h in hist.values() for hi in h}
        if bracket != max(brackets):
            decisions.discard(1)
        meta_.append(
            {
                "decisions": sorted(list(decisions)),
                "n_models": len(hist),
                "bracket": bracket,
                "partial_fit_calls": sum(calls.values()),
                "SuccessiveHalvingSearchCV params": _get_SHA_params(SHAs[bracket]),
            }
        )
    meta_ = sorted(meta_, key=lambda x: x["bracket"])
    return meta_, history_


def _get_SHA_params(SHA):
    """
    Parameters
    ----------
    SHA : SuccessiveHalvingSearchCV

    Returns
    -------
    params : dict
        Dictionary to re-create a SuccessiveHalvingSearchCV without the
        estimator or parameters

    Example
    -------
    >>> from sklearn.linear_model import SGDClassifier
    >>> model = SGDClassifier()
    >>> params = {"alpha": np.logspace(-1, 1)}
    >>> SHA = SuccessiveHalvingSearchCV(model, params, tol=0.1,
    ...                                 patience=True, random_state=42)
    >>> _get_SHA_params(SHA)
    {'aggressiveness': 3,
     'max_iter': 100,
     'n_initial_iter': 9,
     'n_initial_parameters': 10,
     'patience': True,
     'random_state': 42,
     'scoring': None,
     'test_size': None,
     'tol': 0.1}

    """
    return {
        k: v
        for k, v in SHA.get_params().items()
        if "estimator_" not in k and k != "parameters" and k != "estimator"
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
            "n_models": hist["num_estimators"],
            "partial_fit_calls": sum(hist["estimators"].values()),
            "decisions": {int(h) for h in hist["decisions"]},
        }
        for k, hist in hists.items()
    ]
    return info


def _get_patience(patience, max_iter, aggressiveness, tol):
    if not isinstance(patience, bool) and patience < max(max_iter // aggressiveness, 1):
        msg = (
            "The goal of `patience` is to stop training estimators that have "
            "already converged *when few estimators remain*. "
            "Hyperband is already an (almost optimal) adaptive scheme, "
            "and patience should be large to be a minimal layer on top "
            "of Hyperband. \n\n"
            "To clear this warning, set \n\n"
            "    * patience=True\n"
            "    * patience >= {}\n"
            "    * tol=None or tol=np.nan\n\n"
            "instead of patience={} "
        )
        if (tol is not None) and not np.isnan(tol):
            warn(msg.format(max_iter // aggressiveness, patience))
    elif isinstance(patience, bool) and patience:
        return max(max_iter // aggressiveness, 1)
    elif isinstance(patience, bool) and not patience:
        return False
    return int(patience)
