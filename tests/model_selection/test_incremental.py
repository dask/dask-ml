import numpy as np
from sklearn.linear_model import SGDClassifier

from distributed.utils_test import loop, gen_cluster  # noqa: F401
import toolz
from tornado import gen

from dask_ml.datasets import make_classification
from dask_ml.model_selection._incremental import fit, _partial_fit


@gen_cluster(client=True, timeout=None)
def test_basic(c, s, a, b):
    X, y = make_classification(n_samples=10000, n_features=10, chunks=1000)
    model = SGDClassifier(tol=1e-3, penalty='elasticnet')

    params = {'alpha': np.logspace(-2, 1, num=1000),
              'l1_ratio': np.linspace(0, 1, num=1000),
              'average': [True, False]}

    X_test, y_test = X[:1000], y[:1000]
    X_train = X[1000:]
    y_train = y[1000:]

    info, model, history = yield fit(model, params,
                                     X_train, y_train,
                                     X_test, y_test,
                                     start=100,
                                     fit_params={'classes': [0, 1]})

    # Ensure that we touched all data
    keys = {t[0] for t in s.transition_log}
    L = [str(k) in keys for kk in X_train.__dask_keys__() for k in kk]
    assert all(L)

    assert isinstance(model, SGDClassifier)

    while c.futures or s.tasks:  # Cleans up cleanly after running
        yield gen.sleep(0.01)

    # XX_test, yy_test = yield c.compute([X_test, y_test])
    # assert model.score(XX_test, yy_test) == info['score']

    assert len(history) > 200

    groups = toolz.groupby('time_step', history)
    assert (len(groups[0]) > len(groups[1]) >
            len(groups[2]) > len(groups[max(groups)]))
    assert max(groups) > 10


def test_partial_fit_doesnt_mutate_inputs():
    n, d = 100, 20
    X, y = make_classification(
        n_samples=n, n_features=d, random_state=42, chunks=(n, d)
    )
    X = X.compute()
    y = y.compute()
    meta = {
        "iterations": 0,
        "mean_copy_time": 0,
        "mean_fit_time": 0,
        "partial_fit_calls": 1,
    }
    model = SGDClassifier(tol=1e-3)
    model.partial_fit(X[: n // 2], y[: n // 2], classes=np.unique(y))
    new_model, new_meta = _partial_fit(
        (model, meta), X[n // 2:], y[n // 2:],
        fit_params={"classes": np.unique(y)}
    )
    assert meta != new_meta
    assert new_meta["iterations"] == 1
    assert not np.allclose(model.coef_, new_model.coef_)
    assert model.t_ < new_model.t_


@gen_cluster(client=True, timeout=None)
def test_explicit(c, s, a, b):
    X, y = make_classification(n_samples=1000, n_features=10, chunks=(200, 10))
    model = SGDClassifier(tol=1e-3, penalty='elasticnet')
    params = [{'alpha': .1}, {'alpha': .2}]

    def update(scores):
        """ Progress through predefined updates, checking along the way """
        ts = scores[0][-1]['time_step']
        if ts == 0:
            assert len(scores) == len(params)
            assert len(scores[0]) == 1
            assert len(scores[1]) == 1
            return {k: 2 for k in scores}
        if ts == 2:
            assert len(scores) == len(params)
            assert len(scores[0]) == 2
            assert len(scores[1]) == 2
            return {0: 1, 1: 0}
        elif ts == 3:
            assert len(scores) == len(params)
            assert len(scores[0]) == 3
            assert len(scores[1]) == 2
            return {0: 3}
        elif ts == 6:
            assert len(scores) == 1
            assert len(scores[0]) == 4
            return {0: 0}
        else:
            raise Exception()

    info, models, history = yield fit(model, params, X, y,
                                      X.blocks[-1], y.blocks[-1],
                                      update=update, scorer=None,
                                      fit_params={'classes': [0, 1]})
    assert all(model.done() for model in models.values())

    models = yield models
    model, meta = models[0]

    assert meta['params'] == {'alpha': 0.1}
    assert meta['time_step'] == 6
    assert len(models) == len(info) == 1
    assert meta['time_step'] == history[-1]['time_step']

    while s.tasks or c.futures:  # all data clears out
        yield gen.sleep(0.01)
