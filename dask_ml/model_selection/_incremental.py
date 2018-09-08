from __future__ import division

import itertools
import operator
from collections import namedtuple
from copy import deepcopy
from time import time

import dask
import dask.array as da
import numpy as np
import toolz
from dask.distributed import Future, default_client, futures_of, wait
from distributed.utils import log_errors
from sklearn.base import clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from toolz import first
from tornado import gen

from ..utils import check_array
from ._search import _RETURN_TRAIN_SCORE_DEFAULT, DaskBaseSearchCV
from ._split import train_test_split

Results = namedtuple("Results", ["info", "models", "history"])


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
    Results
        A named tuple with three fields: info, models, history

        * info : Dict[model_id, List[Dict]]
            Keys are integers identifying each model. Values are a
            List of Dictk
        * models : Dict[model_id, Future[Estimator]]
            A dictionary with the same keys as `info`. The values
            are futures to the fitted models.
        * history : List[Dict]
            The history of model fitting for each model. Each element
            of the list is a dictionary with the following elements:

            * model_id : int
                A superset of the keys for `info` and `models`.
            * params : Dict[str, Any]
                Parameters this model was trained with.
            * partial_fit_calls : int
                The number of *consecutive* partial fit calls at this stage in
                this models training history.
            * partial_fit_time : float
                Time (in seconds) spent on this partial fit
            * score : float
                Score on the test set for the model at this point in history
            * score_time : float
                Time (in seconds) spent on this scoring.
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
    raise gen.Return(Results(info, models, history))


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


class BaseIncrementalSearch(DaskBaseSearchCV):
    """Base class for estimators using the incremental `fit`.

    Subclasses must implement the following abstract method

    * _additional_calls
    """

    def __init__(
        self,
        estimator,
        parameters,
        test_size=0.15,
        random_state=None,
        scoring=None,
        iid=True,
        refit=True,
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        # TODO: find the subset of sensible parameters.
        self.parameters = parameters
        self.test_size = test_size
        self.random_state = random_state
        super(BaseIncrementalSearch, self).__init__(
            estimator,
            scoring=scoring,
            iid=iid,
            refit=refit,
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
        kwargs["ensure_2d"] = False
        y = check_array(y, **kwargs)
        return X, y

    def _get_train_test_split(self, X, y, **kwargs):
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

    def _additional_calls(self, info):
        raise NotImplementedError

    def _get_params(self):
        """Parameters to pass to `fit`.

        By defualt, a GridSearch over ``self.parameters`` is used.
        """
        return ParameterGrid(self.parameters)

    def _get_history_results(self, results):
        # type: (Results) -> Dict
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
        info, model, history = results
        key = operator.itemgetter("model_id")
        hist2 = sorted(history, key=key)
        return hist2

    def _get_best(self, results, history_results):
        # type: (Dict, Dict) -> Estimator
        """Select the best estimator from the set of estimators."""
        best_model_id = first(results.info)
        key = operator.itemgetter("model_id")
        best_index = 0
        # history_results is sorted by (model_id, partial_fit_calls)
        # best is the model_id with the highest partial fit calls
        for k, v in itertools.groupby(history_results, key=key):
            v = list(v)
            best_index += len(v)
            if k == best_model_id:
                break

        return results.models[best_model_id], best_index

    def _process_results(self, results):
        """Called with the output of `fit` immediately after it finishes.

        Subclasses may update the results here, before further results are
        computed (e.g. ``cv_results_``, ``best_estimator_``).

        By default, results is returned as-is.
        """
        return results

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        X, y = self._check_array(X, y)

        X_train, X_test, y_train, y_test = self._get_train_test_split(X, y)
        scorer = check_scoring(self.estimator, scoring=self.scoring)

        results = yield fit(
            self.estimator,
            self._get_params(),
            X_train,
            y_train,
            X_test,
            y_test,
            additional_calls=self._additional_calls,
            fit_params=fit_params,
            scorer=scorer,
            random_state=self.random_state,
        )
        results = self._process_results(results)
        history_results = self._get_history_results(results)
        best_estimator, best_index = self._get_best(results, history_results)
        best_estimator = yield best_estimator

        # Clean up models we're hanging onto
        ids = list(results.models)
        for model_id in ids:
            del results.models[model_id]

        self.scoring_ = scorer
        self.history_results_ = history_results
        self.best_estimator_ = best_estimator
        self.best_index_ = best_index
        # TODO: More evidence to move away from BaseSearchCV
        # self.best_score_ = self.history_results_[best_index]
        self.n_splits_ = 1
        self.multimetric_ = False  # TODO: is this always true?
        raise gen.Return(self)

    @property
    def best_score_(self):
        check_is_fitted(self, "best_index_")
        return self.history_results_[self.best_index_]["score"]

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model.

        Parameters
        ----------
        X, y : array-like
        **fit_params
            Additional partial fit keyword arguments for the estimator.
        """
        return default_client().sync(self._fit, X, y, **fit_params)


class RandomizedWorstIncrementalSearch(BaseIncrementalSearch):
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
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        self.n_iter = 10
        super(RandomizedWorstIncrementalSearch, self).__init__(
            estimator,
            param_distribution,
            test_size,
            random_state,
            scoring,
            iid,
            refit,
            error_score,
            return_train_score,
            scheduler,
            n_jobs,
            cache_cv,
        )

    def _additional_calls(self, scores):
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

    def _get_params(self):
        return ParameterSampler(self.parameters, self.n_iter, self.random_state)


class RandomIncrementalSearch(BaseIncrementalSearch):
    def __init__(
        self,
        estimator,
        param_distribution,
        n_iter=10,
        max_iter=100,
        patience=10,
        tol=0.001,
        test_size=0.15,
        random_state=None,
        scoring=None,
        iid=True,
        refit=True,
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        self.max_iter = max_iter
        self.n_iter = n_iter
        self.patience = patience
        self.tol = tol
        super(RandomIncrementalSearch, self).__init__(
            estimator,
            param_distribution,
            test_size,
            random_state,
            scoring,
            iid,
            refit,
            error_score,
            return_train_score,
            scheduler,
            n_jobs,
            cache_cv,
        )

    def _get_params(self):
        return ParameterSampler(self.parameters, self.n_iter)

    def _additional_calls(self, info):
        out = {}
        max_iter = self.max_iter
        patience = self.patience
        tol = self.tol

        for ident, records in info.items():
            if max_iter is not None and len(records) > max_iter:
                out[ident] = 0

            elif len(records) > patience:
                old = records[-patience]["score"]
                if all(d["score"] < old + tol for d in records[-patience:]):
                    out[ident] = 0
                else:
                    out[ident] = 1

            else:
                out[ident] = 1
        return out


class SuccessiveReductionSearch(BaseIncrementalSearch):
    def __init__(
        self,
        estimator,
        param_distribution,
        n_iter=10,
        max_iter=100,
        test_size=0.15,
        decay_rate=1.0,
        random_state=None,
        scoring=None,
        iid=True,
        refit=True,
        error_score="raise",
        return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
        scheduler=None,
        n_jobs=-1,
        cache_cv=True,
    ):
        self.max_iter = max_iter
        self.n_iter = n_iter
        self.decay_rate = decay_rate
        super(SuccessiveReductionSearch, self).__init__(
            estimator,
            param_distribution,
            test_size,
            random_state,
            scoring,
            iid,
            refit,
            error_score,
            return_train_score,
            scheduler,
            n_jobs,
            cache_cv,
        )

    def _get_params(self):
        return ParameterSampler(self.parameters, self.n_iter)

    def _additional_calls(self, info):
        def inverse(time):
            """ Decrease target number of models inversely with time """
            return int(self.n_iter / (1 + time) ** self.decay_rate)

        example = toolz.first(info.values())
        time_step = example[-1]["partial_fit_calls"]

        current_time_step = time_step + 1
        next_time_step = current_time_step
        while inverse(current_time_step) == inverse(next_time_step):
            next_time_step += 1

        target = inverse(next_time_step)
        best = toolz.topk(target, info, key=lambda k: info[k][-1]["score"])

        if len(best) == 1:
            [best] = best
            return {best: 0}

        out = {}
        for k in best:
            out[k] = next_time_step - current_time_step

        print(out)
        return out
