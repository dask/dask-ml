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
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted
from toolz import first
from tornado import gen

from ..utils import check_array
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
    scorer : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

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


class BaseIncrementalSearch(BaseEstimator, MetaEstimatorMixin):
    """Base class for estimators using the incremental `fit`.

    Subclasses must implement the following abstract method

    * _additional_calls
    """

    def __init__(
        self, estimator, parameters, test_size=None, random_state=None, scoring=None
    ):
        self.estimator = estimator
        self.parameters = parameters
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring

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
        if self.test_size is None:
            test_size = min(0.2, 1 / X.npartitions)
        else:
            test_size = self.test_size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
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
        best_index = -1
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

    def _check_is_fitted(self, method_name):
        return check_is_fitted(self, "best_estimator_")

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

        self.scorer_ = scorer
        self.history_results_ = history_results
        self.best_estimator_ = best_estimator
        self.best_index_ = best_index
        self.best_score_ = history_results[best_index]["score"]
        self.best_params_ = history_results[best_index]["params"]
        self.n_splits_ = 1
        self.multimetric_ = False  # TODO: is this always true?
        raise gen.Return(self)

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model.

        Parameters
        ----------
        X, y : array-like
        **fit_params
            Additional partial fit keyword arguments for the estimator.
        """
        return default_client().sync(self._fit, X, y, **fit_params)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict(self, X, y=None):
        self._check_is_fitted("predict")
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_proba(self, X):
        self._check_is_fitted("predict_proba")
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def predict_log_proba(self, X):
        self._check_is_fitted("predict_log_proba")
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def decision_function(self, X):
        self._check_is_fitted("decision_function")
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def transform(self, X):
        self._check_is_fitted("transform")
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=("best_estimator_", "estimator"))
    def inverse_transform(self, Xt):
        self._check_is_fitted("inverse_transform")
        return self.best_estimator_.transform(Xt)

    def score(self, X, y=None):
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s" % self.best_estimator_
            )
        return self.scorer_(self.best_estimator_, X, y)


class IncrementalSearch(BaseIncrementalSearch):
    """
    Incrementally search for hyper-parameters on models that support partial_fit

    .. note::

       IncrementalSearch depends on the optional ``distributed`` library.

    This incremental hyper-parameter optimization class starts training the
    model on many hyper-parameters on a small amount of data, and then only
    continues training those models that seem to be performing well.
    The number of actively trained hyper-parameter combinations decays
    with an exponential given by the initial number of parameters and the
    decay rate.

        n_models = n_initial_parameters * (n_batches ** -decay_rate)

    See the :ref:`User Guide <hyperparameter.incremental>` for more.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each initial hyperparameter
        combination. This is assumed to implement the scikit-learn estimator
        interface. Either estimator needs to provide a `score`` function,
        or ``scoring`` must be passed. The estimator must implement
        ``partial_fit``, ``set_params``, and work well with ``clone``.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_initial_parameters : int, default=10
        Number of parameter settings that are sampled.
        This trades off runtime vs quality of the solution.

        Alternatively, you can set this to ``"grid"`` to do a full grid search.

    decay_rate : float, default 1.0
        How quickly to decrease the number partial future fit calls.
        Higher `decay_rate` will result in lower training times, at the cost
        of worse models.

    patience : int, default False
        Maximum number of non-improving scores before we stop training a
        model. Off by default.

    scores_per_fit : int, default 1
        If ``patience`` is used the maximum number of ``partial_fit`` calls
        between ``score`` calls.

    tol : float, default 0.001
        The required level of improvement to consider stopping training on
        that model. The most recent score must be at at most ``tol`` better
        than the all of the previous ``patience`` scores for that model.
        Increasing ``tol`` will tend to reduce training time, at the cost
        of worse models.

    max_iter : int, default 100
        Maximum number of partial fit calls per model.

    test_size : float
        Fraction of the dataset to hold out for computing test scores.
        Defaults to the size of a single partition of the input training set

        .. note::

           The training dataset should fit in memory on a single machine.
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

    Examples
    --------
    Connect to the client and create the data

    >>> from dask.distributed import Client
    >>> client = Client()
    >>> import numpy as np
    >>> from dask_ml.datasets import make_classification
    >>> X, y = make_classification(n_samples=5000000, n_features=20,
    ...                            chunks=100000, random_state=0)

    Our underlying estimator is an SGDClassifier. We specify a few parameters
    common to each clone of the estimator.

    >>> from sklearn.linear_model import SGDClassifier
    >>> model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

    The distribution of parameters we'll sample from.

    >>> params = {'alpha': np.logspace(-2, 1, num=1000),
    ...           'l1_ratio': np.linspace(0, 1, num=1000),
    ...           'average': [True, False]}

    >>> search = IncrementalSearch(model, params, random_state=0)
    >>> search.fit(X, y, classes=[0, 1])
    IncrementalSearch(...)

    Alternatively you can provide keywords to start with more hyper-parameters,
    but stop those that don't seem to improve with more data.

    >>> search = IncrementalSearch(model, params, random_state=0,
    ...                            n_initial_parameters=1000,
    ...                            patience=20, max_iter=100)
    """

    def __init__(
        self,
        estimator,
        param_distribution,
        n_initial_parameters=10,
        decay_rate=1.0,
        test_size=None,
        patience=False,
        tol=0.001,
        scores_per_fit=1,
        max_iter=None,
        random_state=None,
        scoring=None,
    ):
        self.n_initial_parameters = n_initial_parameters
        self.decay_rate = decay_rate
        self.patience = patience
        self.tol = tol
        self.scores_per_fit = scores_per_fit
        self.max_iter = max_iter
        super(IncrementalSearch, self).__init__(
            estimator, param_distribution, test_size, random_state, scoring
        )

    def _get_params(self):
        if self.n_initial_parameters == "grid":
            return ParameterGrid(self.parameters)
        else:
            return ParameterSampler(self.parameters, self.n_initial_parameters)

    def _additional_calls(self, info):
        if self.n_initial_parameters == "grid":
            start = len(ParameterGrid(self.parameters))
        else:
            start = self.n_initial_parameters

        def inverse(time):
            """ Decrease target number of models inversely with time """
            return int(start / (1 + time) ** self.decay_rate)

        example = toolz.first(info.values())
        time_step = example[-1]["partial_fit_calls"]

        current_time_step = time_step + 1
        next_time_step = current_time_step
        while inverse(current_time_step) == inverse(next_time_step) and (
            not self.patience
            or next_time_step - current_time_step < self.scores_per_fit
        ):
            next_time_step += 1

        target = inverse(next_time_step)
        best = toolz.topk(target, info, key=lambda k: info[k][-1]["score"])

        if len(best) == 1:
            [best] = best
            return {best: 0}

        out = {}
        for k in best:
            records = info[k]
            if self.max_iter and len(records) >= self.max_iter:
                out[k] = 0
            elif self.patience and len(records) >= self.patience:
                old = records[-self.patience]["score"]
                if all(d["score"] < old + self.tol for d in records[-self.patience :]):
                    out[k] = 0
                else:
                    out[k] = next_time_step - current_time_step

            else:
                out[k] = next_time_step - current_time_step

        return out
