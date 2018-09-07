from __future__ import division

import itertools
import operator
from copy import deepcopy
from time import time

import dask
import dask.array as da
import numpy as np
from dask.distributed import Future, default_client, futures_of, wait
from distributed.utils import log_errors
from sklearn.base import clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from toolz import first
from tornado import gen

from ..utils import check_array
from ._search import _RETURN_TRAIN_SCORE_DEFAULT, DaskBaseSearchCV
from ._split import train_test_split


def _partial_fit(model_and_meta, X, y, fit_params):
    """
    Call partial_fit on a classifiers with training data X and y

    Arguments
    ---------
    model_and_meta : Tuple[Estimator, dict]
    X, y : np.ndarray, np.ndarray
        Training data
    fit_params : dict
        Extra keyword arguments to pass to partial_fit

    Returns
    -------
    model : Estimator
        The model that has been fit.
    meta : dict
        A new dictionary with updated information.
    """
    with log_errors():
        start = time()
        model, meta = model_and_meta

        if len(X):
            model = deepcopy(model)
            model.partial_fit(X, y, **(fit_params or {}))

        meta = dict(meta)
        meta["partial_fit_calls"] += 1
        meta["partial_fit_time"] = time() - start

        return model, meta


def _score(model_and_meta, X, y, scorer):
    start = time()
    model, meta = model_and_meta
    if scorer:
        score = scorer(model, X, y)
    else:
        score = model.score(X, y)

    meta = dict(meta)
    meta.update(score=score, score_time=time() - start)
    return meta


def _create_model(model, ident, **params):
    """ Create a model by cloning and then setting params """
    with log_errors(pdb=True):
        model = clone(model).set_params(**params)
        return model, {"model_id": ident, "params": params, "partial_fit_calls": 0}


@gen.coroutine
def _fit(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    additional_calls,
    fit_params=None,
    scorer=None,
    random_state=None,
):
    original_model = model
    fit_params = fit_params or {}
    client = default_client()
    rng = check_random_state(random_state)

    info = {}
    models = {}
    scores = {}

    for ident, param in enumerate(params):
        model = client.submit(_create_model, original_model, ident, **param)
        info[ident] = []
        models[ident] = model

    # assume everything in fit_params is small and make it concrete
    fit_params = yield client.compute(fit_params)

    # Convert testing data into a single element on the cluster
    # This assumes that it fits into memory on a single worker
    if isinstance(X_test, da.Array):
        X_test = client.compute(X_test)
    else:
        X_test = yield client.scatter(X_test)
    if isinstance(y_test, da.Array):
        y_test = client.compute(y_test)
    else:
        y_test = yield client.scatter(y_test)

    # Convert to batches of delayed objects of numpy arrays
    X_train, y_train = dask.persist(X_train, y_train)
    X_train = sorted(futures_of(X_train), key=lambda f: f.key)
    y_train = sorted(futures_of(y_train), key=lambda f: f.key)
    assert len(X_train) == len(y_train)

    # Order by which we process training data futures
    order = []

    def get_futures(partial_fit_calls):
        """ Policy to get training data futures

        Currently we compute once, and then keep in memory.
        Presumably in the future we'll want to let data drop and recompute.
        This function handles that policy internally, and also controls random
        access to training data.
        """
        # Shuffle blocks going forward to get uniform-but-random access
        while partial_fit_calls >= len(order):
            L = list(range(len(X_train)))
            rng.shuffle(L)
            order.extend(L)
        j = order[partial_fit_calls]
        return X_train[j], y_train[j]

    # Submit initial partial_fit and score computations on first batch of data
    X_future, y_future = get_futures(0)
    X_future_2, y_future_2 = get_futures(1)
    _models = {}
    _scores = {}
    _specs = {}

    d_partial_fit = dask.delayed(_partial_fit)
    d_score = dask.delayed(_score)
    for ident, model in models.items():
        model = d_partial_fit(model, X_future, y_future, fit_params)
        score = d_score(model, X_test, y_test, scorer)
        spec = d_partial_fit(model, X_future_2, y_future_2, fit_params)
        _models[ident] = model
        _scores[ident] = score
        _specs[ident] = spec
    _models, _scores, _specs = dask.persist(
        _models, _scores, _specs, priority={tuple(_specs.values()): -1}
    )
    _models = {k: list(v.dask.values())[0] for k, v in _models.items()}
    _scores = {k: list(v.dask.values())[0] for k, v in _scores.items()}
    _specs = {k: list(v.dask.values())[0] for k, v in _specs.items()}
    models.update(_models)
    scores.update(_scores)
    speculative = _specs

    new_scores = list(_scores.values())
    history = []

    # async for future, result in seq:
    while True:
        metas = yield client.gather(new_scores)

        for meta in metas:
            ident = meta["model_id"]

            info[ident].append(meta)
            history.append(meta)

        instructions = additional_calls(info)
        bad = set(models) - set(instructions)

        # Delete the futures of bad models.  This cancels speculative tasks
        for ident in bad:
            del models[ident]
            del scores[ident]
            del info[ident]

        if not any(instructions.values()):
            break

        _models = {}
        _scores = {}
        _specs = {}
        for ident, k in instructions.items():
            start = info[ident][-1]["partial_fit_calls"] + 1
            if k:
                k -= 1
                model = speculative.pop(ident)
                for i in range(k):
                    X_future, y_future = get_futures(start + i)
                    model = d_partial_fit(model, X_future, y_future, fit_params)
                score = d_score(model, X_test, y_test, scorer)
                X_future, y_future = get_futures(start + k)
                spec = d_partial_fit(model, X_future, y_future, fit_params)
                _models[ident] = model
                _scores[ident] = score
                _specs[ident] = spec

        _models2, _scores2, _specs2 = dask.persist(
            _models, _scores, _specs, priority={tuple(_specs.values()): -1}
        )
        _models2 = {
            k: v if isinstance(v, Future) else list(v.dask.values())[0]
            for k, v in _models2.items()
        }

        _scores2 = {k: list(v.dask.values())[0] for k, v in _scores2.items()}
        _specs2 = {k: list(v.dask.values())[0] for k, v in _specs2.items()}
        models.update(_models2)
        scores.update(_scores2)
        speculative = _specs2

        new_scores = list(_scores2.values())

    models = {k: client.submit(operator.getitem, v, 0) for k, v in models.items()}
    yield wait(models)
    raise gen.Return((info, models, history))


def fit(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    additional_calls,
    fit_params=None,
    scorer=None,
    random_state=None,
):
    """ Find a good model and search among a space of hyper-parameters

    This does a hyper-parameter search by creating many models and then fitting
    them incrementally on batches of data and reducing the number of models based
    on the scores computed during training.  Over time fewer and fewer models
    remain.  We train these models for increasingly long times.

    The model, number of starting parameters, and decay can all be provided as
    configuration parameters.

    Training data should be given as Dask arrays.  It can be large.  Testing
    data should be given either as a small dask array or as a numpy array.  It
    should fit on a single worker.

    Parameters
    ----------
    model : Estimator
    params : List[Dict]
        Parameters to start training on model
    X_train : dask Array
    y_train : dask Array
    X_test : Array
        Numpy array or small dask array.  Should fit in single node's memory.
    y_test : Array
        Numpy array or small dask array.  Should fit in single node's memory.
    additional_calls : callable
        A function that takes information about scoring history per model and
        returns the number of additional partial fit calls to run on each model
    fit_params : dict
        Extra parameters to give to partial_fit
    scorer :
    random_state :

    Examples
    --------
    >>> import numpy as np
    >>> from dask_ml.datasets import make_classification
    >>> X, y = make_classification(n_samples=5000000, n_features=20,
    ...                            chunks=100000, random_state=0)

    >>> from sklearn.linear_model import SGDClassifier
    >>> model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

    >>> from sklearn.model_selection import ParameterSampler
    >>> params = {'alpha': np.logspace(-2, 1, num=1000),
    ...           'l1_ratio': np.linspace(0, 1, num=1000),
    ...           'average': [True, False]}
    >>> params = list(ParameterSampler(params, 10, random_state=0))

    >>> X_test, y_test = X[:100000], y[:100000]
    >>> X_train = X[100000:]
    >>> y_train = y[100000:]

    >>> def remove_worst(scores):
    ...    last_score = {model_id: info[-1]['score']
    ...                  for model_id, info in scores.items()}
    ...    worst_score = min(last_score.values())
    ...    out = {}
    ...    for model_id, score in last_score.items():
    ...        if score != worst_score:
    ...            out[model_id] = 1  # do one more training step
    ...    if len(out) == 1:
    ...        out = {k: 0 for k in out}  # no more work to do, stops execution
    ...    return out

    >>> from dask.distributed import Client
    >>> client = Client(processes=False)

    >>> from dask_ml.model_selection._incremental import fit
    >>> info, models, history = fit(model, params,
    ...                             X_train, y_train,
    ...                             X_test, y_test,
    ...                             additional_calls=remove_worst,
    ...                             fit_params={'classes': [0, 1]},
    ...                             random_state=0)

    >>> models
    {2: <Future: status: finished, type: SGDClassifier, key: ...}
    >>> models[2].result()
    SGDClassifier(...)
    >>> info[2][-1]  # doctest: +SKIP
    {'model_id': 2,
     'params': {'l1_ratio': 0.9529529529529529, 'average': False,
                'alpha': 0.014933932161242525},
     'partial_fit_calls': 8,
     'partial_fit_time': 0.17334818840026855,
     'score': 0.58765,
     'score_time': 0.031442880630493164}

    Returns
    -------
    info : Dict[int, List[Dict]]
        Scoring history of each successful model, keyed by model ID.
        This has the parameters, scores, and timing information over time
    models : Dict[int, Future]
        Dask futures pointing to trained models
    history : List[Dict]
        A history of all models scores over time
    """
    return default_client().sync(
        _fit,
        model,
        params,
        X_train,
        y_train,
        X_test,
        y_test,
        additional_calls,
        fit_params=fit_params,
        scorer=scorer,
        random_state=random_state,
    )


# ----------------------------------------------------------------------------
# Base class for scikit-learn compatible estimators using fit
# ----------------------------------------------------------------------------


class BaseIncrementalSearchCV(DaskBaseSearchCV):
    """Base class for estimators using the incremental `fit`.

    Subclasses must implement the following abstract method

    * _get_ids_and_args : Build the dict of
       ``{model_id : (param_list, additional_calls)}``
        that are eventually passed to `fit`
    """

    def __init__(
        self,
        estimator,
        params,
        test_size=0.15,
        random_state=None,
        scoring=None,
        iid=True,
        refit=True,
        cv=None,
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        # TODO: find the subset of sensible parameters.
        self.params = params
        self.test_size = test_size
        self.random_state = random_state
        super(BaseIncrementalSearchCV, self).__init__(
            estimator,
            scoring=scoring,
            iid=iid,
            refit=refit,
            cv=cv,
            error_score=error_score,
            return_train_score=return_train_score,
            scheduler=scheduler,
            n_jobs=n_jobs,
            cache_cv=cache_cv,
        )

    def _check_array(self, X, y, **kwargs):
        """Validate the data arguments X and y.

        By default, NumPy arrays are converted to 1-block dask arrays.

        Parameters
        ----------
        X, y : array-like
        """
        if isinstance(X, np.ndarray):
            X = da.from_array(X, X.shape)
        if isinstance(y, np.ndarray):
            y = da.from_array(y, y.shape)
        X = check_array(X, **kwargs)
        y = check_array(y, **kwargs)
        return X, y

    def _get_cv(self, X, y, **kwargs):
        """CV-Split the arrays X and y

        By default, :meth:`dask_ml.model_selection.train_test_split`
        is used with ``self.test_size``. The test set is expected to
        fit in memory on each worker machine.

        Parameters
        ----------
        X, y : dask.array.Array
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )
        return X_train, X_test, y_train, y_test

    def _get_ids_and_args(self):
        # type: () -> Dict[int, Tuple(List[Dict], Union[Callable, object])]
        """Abstract method for generating the argumnets passed to `fit`.

        Returns
        -------
        ids_and_args : Dict
            Should have integers for keys and the following arguments

            * param_list : List[Dict]
            * additional_fit_calls : Union[Callable, object]
                Objects with a ``fit`` method have the bound
                ``additional_fit_calls.fit`` method passed through to `fit`.
                (Other) callables are simply passed through.
        """
        raise NotImplementedError

    def _get_cv_results(self, results):
        # type: (Dict) -> Dict
        """Construct the CV results.

        Has the following keys:

        * params
        * test_score
        * mean_test_score
        * rank_test_score
        * mean_partial_fit_time
        * std_partial_fit_time
        * mean_score_time
        * std_score_time
        * partial_fit_calls
        * model_id
        """
        info, model, history = zip(*results.values())
        info = list(itertools.chain.from_iterable(x.values() for x in info))
        model = list(itertools.chain.from_iterable(x.values() for x in model))
        history = list(itertools.chain.from_iterable(history))

        scores = np.array([x["score"] for v in info for x in v])
        ranks = np.argsort(-1 * scores) + 1
        assert max(scores) == scores[ranks - 1][0]
        model_ids = []
        for loop_id in results:
            loop_info = results[loop_id][0]
            model_ids.extend(
                [
                    "{}-{}".format(loop_id, x["model_id"])
                    for v in loop_info.values()
                    for x in v
                ]
            )

        cv_results = {
            "params": [x["params"] for v in info for x in v],
            "test_score": scores,
            "mean_test_score": scores,  # for sklearn comptability
            "rank_test_score": ranks,
            "mean_partial_fit_time": np.array(
                [np.mean(x["partial_fit_time"]) for v in info for x in v]
            ),
            "std_partial_fit_time": np.array(
                [np.std(x["partial_fit_time"]) for v in info for x in v]
            ),
            "mean_score_time": np.array(
                [np.mean(x["score_time"]) for v in info for x in v]
            ),
            "std_score_time": np.array(
                [np.std(x["score_time"]) for v in info for x in v]
            ),
            "partial_fit_calls": [x["partial_fit_calls"] for v in info for x in v],
            "model_id": model_ids,
        }
        return cv_results

    def _get_best(self, results, cv_results):
        # type: (Dict, Dict) -> Estimator
        """Select the best estimator from the set of estimators."""
        rank = cv_results["rank_test_score"]
        best_index = np.arange(len(rank))[rank - 1][0]
        best_model_id = cv_results["model_id"][best_index]
        assert cv_results["test_score"][best_index] == max(cv_results["test_score"])

        loop_id, model_id = map(int, best_model_id.split("-"))
        return results[loop_id][1][model_id].result(), best_index

    def _update_results(self, results):
        # type: (Dict) -> None
        """Update `self` with attributes extracted from `results`."""
        pass

    def _process_results(self, results):
        """Called with the output of `fit` immediately after it finishes.

        Subclasses may update the results here, before further results are
        computed (e.g. ``cv_results_``, ``best_estimator_``).

        By default, results is returned as-is.
        """
        return results

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model

        Parameters
        ----------
        X, y : array-like
        **fit_params
            Additional partial fit keyword arguments for the estimator.
        """
        ids_and_args = self._get_ids_and_args()
        X, y = check_array(X, y)

        X_train, X_test, y_train, y_test = self._get_cv(X, y)
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        results = {
            model_id: fit(
                self.estimator,
                param_list,
                X_train,
                y_train,
                X_test,
                y_test,
                additional_calls=getattr(additional_calls, "fit", additional_calls),
                fit_params=fit_params,
                scorer=scorer,
                random_state=self.random_state,
            )
            for model_id, (param_list, additional_calls) in ids_and_args.items()
        }
        results = self._process_results(results)
        self._update_results(results)
        cv_results = self._get_cv_results(results)
        best_estimator, best_index = self._get_best(results, cv_results)

        # Clean up models we're hanging onto
        ids = [
            (loop_id, model_id)
            for loop_id in results
            for model_id in results[loop_id][1]
        ]
        for (loop_id, model_id) in ids:
            del results[loop_id][1][model_id]

        self.scoring_ = scorer
        self.cv_results_ = cv_results
        self.best_estimator_ = best_estimator
        self.best_index_ = best_index
        self.n_splits_ = 1
        self.multimetric_ = False  # TODO: is this always true?
        return self


class IncrementalRandomizedSearchCV(BaseIncrementalSearchCV):
    def __init__(
        self,
        estimator,
        param_distribution,
        n_iter=10,
        test_size=0.15,
        random_state=None,
        scoring=None,
        iid=True,
        refit=True,
        cv=None,
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        self.n_iter = 10
        super(IncrementalRandomizedSearchCV, self).__init__(
            estimator,
            param_distribution,
            test_size,
            random_state,
            scoring,
            iid,
            refit,
            cv,
            error_score,
            return_train_score,
            scheduler,
            n_jobs,
            cache_cv,
        )

    def _get_ids_and_args(self):
        return {0: (ParameterSampler(self.params, self.n_iter), remove_worst)}

    def _update_results(self, results):
        self.info_, _, self.history_ = results[0]


def remove_worst(scores):
    # type: Dict[Int, Dict] -> Dict[Int, Int]
    """Default `additional_calls` strategy for IncrementalSearch.

    Removes the lowest scoring model from a batch of models.
    """
    last_score = {model_id: info[-1]["score"] for model_id, info in scores.items()}
    worst_score = min(last_score.values())
    out = {}
    for model_id, score in last_score.items():
        if score != worst_score:
            out[model_id] = 1  # do one more training step

    if len(out) == 0:
        # we have a tie where each model is equal to the worst.
        # Arbitrarily pick the first.
        out = {first(last_score): 0}
    elif len(out) == 1:
        out = {k: 0 for k in out}  # no more work to do, stops execution
    return out
