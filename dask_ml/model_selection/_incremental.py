from __future__ import division

import operator
from copy import deepcopy
from time import time

import dask
import dask.array as da
from dask.distributed import Future, default_client, futures_of, wait
from distributed.utils import log_errors
from sklearn.base import clone
from sklearn.utils import check_random_state
from tornado import gen


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
