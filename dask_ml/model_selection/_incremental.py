from __future__ import division

from collections import defaultdict
from copy import deepcopy
import functools

from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from tornado import gen
import toolz

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
    with log_errors(pdb=True):
        model, meta = model_and_meta

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


def inverse(start, batch):
    """ Decrease target number of models inversely with time

    This means that we train many models to start for a brief time and a few
    models at the end for a long time
    """
    return int(start / (1 + batch))


@gen.coroutine
def _fit(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    start=1000,
    fit_params=None,
    random_state=None,
    scorer=None,
    target=inverse,
):
    original_model = model
    fit_params = fit_params or {}
    client = default_client()
    rng = check_random_state(random_state)
    param_iterator = iter(ParameterSampler(params, start, random_state=rng))
    target = functools.partial(target, start)

    info = {}
    models = {}
    scores = {}

    for ident in range(start):
        params = next(param_iterator)
        model = client.submit(_create_model, original_model, ident,
                              random_state=rng.randint(2**31), **params)
        info[ident] = {'params': params, 'param_index': ident}
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

    # Create order by which we process batches
    # TODO: create a non-repetitive random and uniform ordering
    order = list(range(len(X_train)))
    rng.shuffle(order)
    seen = {}
    tokens = {}

    def get_futures(time_step):
        j = order[time_step % len(order)]

        if time_step < len(order) and j not in seen:  # new future, need to tell scheduler about it
            X_future = client.compute(X_train[j])
            y_future = client.compute(y_train[j])
            seen[j] = (X_future.key, y_future.key)

            # This is a hack to keep the futures in the scheduler but not in memory
            X_token = client.submit(len, X_future)
            y_token = client.submit(len, y_future)
            tokens[time_step] = (X_token, y_token)

            return X_future, y_future

        else:
            x_key, y_key = seen[j]
            return Future(x_key), Future(y_key)

    # Submit initial partial_fit and score computations on first batch of data
    X_future, y_future = get_futures(0)
    for ident, model in models.items():
        model = client.submit(_partial_fit, model, X_future, y_future, fit_params)
        score = client.submit(_score, model, X_test, y_test, scorer)
        models[ident] = model
        scores[ident] = score

    done = defaultdict(set)
    seq = as_completed(scores.values(), with_results=True)
    current_time_step = 0
    next_time_step = current_time_step + 1
    optimistic = set()  # set of fits that we might or might not want to keep
    history = []

    # async for future, result in seq:
    while not seq.is_empty():
        future, meta = yield seq.__anext__()
        if future.cancelled():
            continue
        time_step = meta['time_step']
        ident = meta['ident']

        done[time_step].add(ident)
        info[ident].update(meta)
        history.append(meta)

        # Evolve the model by a few time steps, then call score on the last one
        model = models[ident]
        for i in range(time_step, next_time_step):
            X_future, y_future = get_futures(i + 1)
            model = client.submit(_partial_fit, model, X_future, y_future,
                                  fit_params, priority=-i + meta['score'])
        score = client.submit(_score, model, X_test, y_test, scorer,
                              priority=-time_step + meta['score'])
        models[ident] = model
        scores[ident] = score
        optimistic.add(ident)  # we're not yet sure that we want to do this

        # We've now finished a full set of models
        # It's time to select the ones that get to survive and remove the rest
        if time_step == current_time_step and len(done[time_step]) >= len(models):

            # Step forward in time until we'll want to contract models again
            current_time_step = next_time_step
            next_time_step = current_time_step + 1
            while target(current_time_step) == target(next_time_step):
                next_time_step += 1

            # Select the best models by score
            good = set(toolz.topk(target(current_time_step), models, key=lambda i: info[i]['score']))
            bad = set(models) - good

            # Delete the futures of the other models.  This cancels optimistically submitted tasks
            for ident in bad:
                del models[ident]
                del scores[ident]

            # Add back into the as_completed iterator
            for ident in optimistic & good:
                seq.add(scores[ident])
            optimistic.clear()

            assert len(models) == target(current_time_step)

            if len(good) == 1:  # found the best one?  Break.
                break

    [best] = good
    model, meta = yield models[best]
    raise gen.Return((info[best], model, history))


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
