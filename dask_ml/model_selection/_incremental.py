from __future__ import division

from copy import deepcopy

from sklearn.base import clone
from sklearn.utils import check_random_state
from tornado import gen

import dask
import dask.array as da
from dask.distributed import as_completed, default_client, Future
from distributed.utils import log_errors


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
        model, meta = model_and_meta

        if len(X):
            model = deepcopy(model)
            model.partial_fit(X, y, **(fit_params or {}))

        meta = dict(meta)
        meta['time_step'] += 1

        return model, meta


def _score(model_and_meta, X, y, scorer):
    model, meta = model_and_meta
    if scorer:
        score = scorer(model, X, y)
    else:
        score = model.score(X, y)

    meta = dict(meta)
    meta.update(score=score)
    return meta


def _create_model(model, ident, **params):
    """ Create a model by cloning and then setting params """
    with log_errors(pdb=True):
        model = clone(model).set_params(**params)
        return model, {'ident': ident, 'params': params, 'time_step': -1}


@gen.coroutine
def _fit(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    update,
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
        y_test = yield client.scatter(y_test)
    if isinstance(y_test, da.Array):
        y_test = client.compute(y_test)
    else:
        y_test = yield client.scatter(y_test)

    # Convert to batches of delayed objects of numpy arrays
    X_train = X_train.to_delayed()
    if hasattr(X_train, 'squeeze'):
        X_train = X_train.squeeze()
    y_train = y_train.to_delayed()
    if hasattr(y_train, 'squeeze'):
        y_train = y_train.squeeze()
    X_train, y_train = dask.optimize(X_train.tolist(), y_train.tolist())

    # Order by which we process training data futures
    order = []
    seen = {}
    tokens = {}

    def get_futures(time_step):
        """ Policy to get training data futures

        Currently we compute once, and then keep in memory.
        Presumably in the future we'll want to let data drop and recompute.
        This function handles that policy internally, and also controls random
        access to training data.
        """
        # Shuffle blocks going forward to get uniform-but-random access
        while time_step >= len(order):
            L = list(range(len(X_train)))
            rng.shuffle(L)
            order.extend(L)

        j = order[time_step]

        if j in seen:
            x_key, y_key = seen[j]
            return Future(x_key, inform=False), Future(y_key, inform=False)
        else:
            # new future, need to tell scheduler about it
            X_future = client.compute(X_train[j])
            y_future = client.compute(y_train[j])
            seen[j] = (X_future.key, y_future.key)

            # Hack to keep the futures in the scheduler but not in memory
            X_token = client.submit(len, X_future)
            y_token = client.submit(len, y_future)
            tokens[j] = (X_token, y_token)
            return X_future, y_future

    # Submit initial partial_fit and score computations on first batch of data
    X_future, y_future = get_futures(0)
    for ident, model in models.items():
        model = client.submit(_partial_fit, model, X_future, y_future, fit_params)
        score = client.submit(_score, model, X_test, y_test, scorer)
        models[ident] = model
        scores[ident] = score

    seq = as_completed(scores.values(), with_results=True)
    speculative = dict()  # models that we might or might not want to keep
    history = []
    number_to_complete = len(models)

    # async for future, result in seq:
    while not seq.is_empty():
        future, meta = yield seq.__anext__()
        if future.cancelled():
            continue
        ident = meta['ident']

        info[ident].append(meta)
        history.append(meta)

        # Evolve the model one more step
        model = models[ident]
        X_future, y_future = get_futures(meta['time_step'] + 1)
        model = client.submit(_partial_fit, model, X_future, y_future,
                              fit_params)
        speculative[ident] = model

        # Have we finished a full set of models?
        if len(speculative) == number_to_complete:
            instructions = update(info)

            bad = set(models) - set(instructions)

            # Delete the futures of bad models.  This cancels speculative tasks
            for ident in bad:
                del models[ident]
                del scores[ident]
                del info[ident]

            if not any(instructions.values()):
                break

            for ident, k in instructions.items():
                start = info[ident][-1]['time_step'] + 1
                if k:
                    if ident in speculative:
                        model = speculative.pop(ident)
                        k -= 1
                    else:
                        model = models[ident]
                    for i in range(k):
                        X_future, y_future = get_futures(start + i)
                        model = client.submit(_partial_fit, model, X_future, y_future, fit_params)
                    score = client.submit(_score, model, X_test, y_test, scorer)
                    models[ident] = model
                    scores[ident] = score

                    seq.add(score)

            number_to_complete = len([v for v in instructions.values() if v])

            speculative.clear()

    raise gen.Return((info, models, history))


def fit(*args, **kwargs):
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
    params : dict
        parameter grid to be given to ParameterSampler
    X_train : dask Array
    y_train : dask Array
    X_test : Array
        Numpy array or small dask array.  Should fit in memory.
    y_test : Array
        Numpy array or small dask array.  Should fit in memory.
    start : int
        Number of parameters to start with
    fit_params : dict
        Extra parameters to give to partial_fit
    random_state :
    scorer :
    target : callable
        A function that takes the start value and the current time step and
        returns the number of desired models at that time step

    Examples
    --------
    >>> X, y = make_classification(n_samples=5000000, n_features=20,
    ...                            chunks=100000)

    >>> model = SGDClassifier(tol=1e-3, penalty='elasticnet')
    >>> params = {'alpha': np.logspace(-2, 1, num=1000),
    ...           'l1_ratio': np.linspace(0, 1, num=1000),
    ...           'average': [True, False]}

    >>> X_test, y_test = X[:100000], y[:100000]
    >>> X_train = X[100000:]
    >>> y_train = y[100000:]

    >>> info, model, history = yield fit(model, params,
    ...                                  X_train, y_train,
    ...                                  X_test, y_test,
    ...                                  start=100,
    ...                                  fit_params={'classes': [0, 1]})
    """
    return default_client().sync(_fit, *args, **kwargs)
