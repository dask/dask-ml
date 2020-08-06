import asyncio
import concurrent.futures
import itertools
import logging
import math
import random
import sys

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import scipy
import toolz
from dask.distributed import Future
from distributed.utils_test import (  # noqa: F401
    captured_logger,
    cluster,
    gen_cluster,
    loop,
)
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils import check_random_state

from dask_ml._compat import DISTRIBUTED_2_5_0
from dask_ml.datasets import make_classification
from dask_ml.model_selection import (
    HyperbandSearchCV,
    IncrementalSearchCV,
    InverseDecaySearchCV,
)
from dask_ml.model_selection._incremental import _partial_fit, _score, fit
from dask_ml.model_selection.utils_test import LinearFunction, _MaybeLinearFunction
from dask_ml.utils import ConstantFunction

pytestmark = [
    pytest.mark.skipif(not DISTRIBUTED_2_5_0, reason="hangs"),
    pytest.mark.filterwarnings("ignore:decay_rate"),
]  # decay_rate warnings are tested in test_incremental_warns.py


@gen_cluster(client=True, timeout=1000)
async def test_basic(c, s, a, b):
    def _additional_calls(info):
        pf_calls = {k: v[-1]["partial_fit_calls"] for k, v in info.items()}
        ret = {k: int(calls < 10) for k, calls in pf_calls.items()}
        if len(ret) == 1:
            return {list(ret)[0]: 0}

        # Don't train one model (but keep model 0)
        some_keys = set(ret.keys()) - {0}
        key_to_drop = random.choice(list(some_keys))
        return {k: v for k, v in ret.items() if k != key_to_drop}

    X, y = make_classification(n_samples=1000, n_features=5, chunks=100)
    model = ConstantFunction()

    params = {"value": uniform(0, 1)}

    X_test, y_test = X[:100], y[:100]
    X_train = X[100:]
    y_train = y[100:]

    n_parameters = 5
    param_list = list(ParameterSampler(params, n_parameters))

    info, models, history, best = await fit(
        model,
        param_list,
        X_train,
        y_train,
        X_test,
        y_test,
        _additional_calls,
        fit_params={"classes": [0, 1]},
    )

    # Ensure that we touched all data
    keys = {t[0] for t in s.transition_log}
    L = [str(k) in keys for kk in X_train.__dask_keys__() for k in kk]
    assert all(L)

    for model in models.values():
        assert isinstance(model, Future)
        model2 = await model
        assert isinstance(model2, ConstantFunction)

    XX_test = await c.compute(X_test)
    yy_test = await c.compute(y_test)
    model = await models[0]
    assert model.score(XX_test, yy_test) == info[0][-1]["score"]

    # `<` not `==` because we randomly dropped one model every iteration
    assert len(history) < n_parameters * 10
    for h in history:
        assert {
            "partial_fit_time",
            "score_time",
            "score",
            "model_id",
            "params",
            "partial_fit_calls",
        }.issubset(set(h.keys()))

    groups = toolz.groupby("partial_fit_calls", history)
    assert len(groups[1]) > len(groups[2]) > len(groups[3]) > len(groups[max(groups)])
    assert max(groups) == n_parameters

    keys = list(models.keys())
    for key in keys:
        del models[key]

    while c.futures or s.tasks:  # Make sure cleans up cleanly after running
        await asyncio.sleep(0.1)

    # smoke test for ndarray X_test and y_test
    X_test = await c.compute(X_test)
    y_test = await c.compute(y_test)
    info, models, history, best = await fit(
        model,
        param_list,
        X_train,
        y_train,
        X_test,
        y_test,
        _additional_calls,
        fit_params={"classes": [0, 1]},
    )
    assert True  # smoke test to make sure reached


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
        "partial_fit_calls": 0,
    }
    model = SGDClassifier(tol=1e-3)
    model.partial_fit(X[: n // 2], y[: n // 2], classes=np.unique(y))
    new_model, new_meta = _partial_fit(
        (model, meta), X[n // 2 :], y[n // 2 :], fit_params={"classes": np.unique(y)}
    )
    assert meta != new_meta
    assert new_meta["partial_fit_calls"] == 1
    assert not np.allclose(model.coef_, new_model.coef_)
    assert model.t_ < new_model.t_
    assert new_meta["partial_fit_time"] >= 0
    new_meta2 = _score((model, new_meta), X[n // 2 :], y[n // 2 :], None)
    assert new_meta2["score_time"] >= 0
    assert new_meta2 != new_meta


@gen_cluster(client=True)
async def test_explicit(c, s, a, b):
    X, y = make_classification(n_samples=1000, n_features=10, chunks=(200, 10))
    model = SGDClassifier(tol=1e-3, penalty="elasticnet")
    params = [{"alpha": 0.1}, {"alpha": 0.2}]

    def additional_calls(scores):
        """ Progress through predefined updates, checking along the way """
        ts = scores[0][-1]["partial_fit_calls"]
        ts -= 1  # partial_fit_calls = time step + 1
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

    info, models, history, best = await fit(
        model,
        params,
        X,
        y,
        X.blocks[-1],
        y.blocks[-1],
        additional_calls,
        scorer=None,
        fit_params={"classes": [0, 1]},
    )
    assert all(model.done() for model in models.values())

    models = await c.compute(models)
    model = models[0]
    meta = info[0][-1]

    assert meta["params"] == {"alpha": 0.1}
    assert meta["partial_fit_calls"] == 6 + 1
    assert len(info) > len(models) == 1
    assert set(models.keys()).issubset(set(info.keys()))
    assert meta["partial_fit_calls"] == history[-1]["partial_fit_calls"]
    calls = {k: [h["partial_fit_calls"] for h in hist] for k, hist in info.items()}
    for k, call in calls.items():
        assert (np.diff(call) >= 1).all()
    assert set(models.keys()) == {0}
    del models[0]

    while s.tasks or c.futures:  # all data clears out
        await asyncio.sleep(0.1)


@gen_cluster(client=True)
async def test_search_basic(c, s, a, b):
    for decay_rate, input_type, memory in itertools.product(
        {0, 1}, ["array", "dataframe"], ["distributed"]
    ):
        success = await _test_search_basic(decay_rate, input_type, memory, c, s, a, b)
        assert isinstance(success, bool) and success, "Did the test run?"


async def _test_search_basic(decay_rate, input_type, memory, c, s, a, b):
    X, y = make_classification(n_samples=1000, n_features=5, chunks=(100, 5))
    assert isinstance(X, da.Array)
    if memory == "distributed" and input_type == "dataframe":
        X = dd.from_array(X)
        y = dd.from_array(y)
        assert isinstance(X, dd.DataFrame)
    elif memory == "local":
        X, y = await c.compute([X, y])
        assert isinstance(X, np.ndarray)
        if input_type == "dataframe":
            X, y = pd.DataFrame(X), pd.DataFrame(y)
            assert isinstance(X, pd.DataFrame)

    model = SGDClassifier(tol=1e-3, loss="log", penalty="elasticnet")

    params = {"alpha": np.logspace(-2, 2, 100), "l1_ratio": np.linspace(0.01, 1, 200)}

    kwargs = dict(n_initial_parameters=20, max_iter=10)
    if decay_rate == 0:
        search = IncrementalSearchCV(model, params, **kwargs)
    elif decay_rate == 1:
        search = InverseDecaySearchCV(model, params, **kwargs)
    else:
        raise ValueError()
    await search.fit(X, y, classes=[0, 1])

    assert search.history_
    for d in search.history_:
        assert d["partial_fit_calls"] <= search.max_iter + 1
    assert isinstance(search.best_estimator_, SGDClassifier)
    assert search.best_score_ > 0
    assert "visualize" not in search.__dict__
    assert search.best_params_
    assert search.cv_results_ and isinstance(search.cv_results_, dict)
    assert {
        "mean_partial_fit_time",
        "mean_score_time",
        "std_partial_fit_time",
        "std_score_time",
        "test_score",
        "rank_test_score",
        "model_id",
        "params",
        "partial_fit_calls",
        "param_alpha",
        "param_l1_ratio",
    }.issubset(set(search.cv_results_.keys()))
    assert len(search.cv_results_["param_alpha"]) == 20

    assert all(isinstance(v, np.ndarray) for v in search.cv_results_.values())
    if decay_rate == 0:
        assert (
            search.cv_results_["test_score"][search.best_index_]
            >= search.cv_results_["test_score"]
        ).all()
        assert search.cv_results_["rank_test_score"][search.best_index_] == 1
    else:
        assert all(search.cv_results_["test_score"] >= 0)
        assert all(search.cv_results_["rank_test_score"] >= 1)
    assert all(search.cv_results_["partial_fit_calls"] >= 1)
    assert len(np.unique(search.cv_results_["model_id"])) == len(
        search.cv_results_["model_id"]
    )
    assert sorted(search.model_history_.keys()) == list(range(20))
    assert set(search.model_history_[0][0].keys()) == {
        "model_id",
        "params",
        "partial_fit_calls",
        "partial_fit_time",
        "score",
        "score_time",
        "elapsed_wall_time",
    }

    # Dask Objects are lazy
    X_ = await c.compute(X)

    proba = search.predict_proba(X)
    log_proba = search.predict_log_proba(X)
    assert proba.shape[1] == 2
    assert proba.shape[0] == 1000 or math.isnan(proba.shape[0])
    assert log_proba.shape[1] == 2
    assert log_proba.shape[0] == 1000 or math.isnan(proba.shape[0])

    assert isinstance(proba, da.Array)
    assert isinstance(log_proba, da.Array)

    proba_ = search.predict_proba(X_)
    log_proba_ = search.predict_log_proba(X_)

    da.utils.assert_eq(proba, proba_)
    da.utils.assert_eq(log_proba, log_proba_)

    decision = search.decision_function(X_)
    assert decision.shape == (1000,) or math.isnan(decision.shape[0])
    return True


@gen_cluster(client=True)
def test_search_plateau_patience(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))

    class ConstantClassifier(SGDClassifier):
        def __init__(self, value=0):
            self.value = value
            super(ConstantClassifier, self).__init__(tol=1e-3)

        def score(self, *args, **kwargs):
            return self.value

    params = {"value": np.random.rand(10)}
    model = ConstantClassifier()

    search = IncrementalSearchCV(
        model, params, n_initial_parameters=10, patience=5, tol=0, max_iter=10,
    )
    yield search.fit(X, y, classes=[0, 1])

    assert search.history_
    assert pd.DataFrame(search.history_).partial_fit_calls.max() <= 5
    assert isinstance(search.best_estimator_, SGDClassifier)
    assert search.best_score_ == params["value"].max() == search.best_estimator_.value
    assert "visualize" not in search.__dict__
    assert search.best_score_ > 0

    X_test, y_test = yield c.compute([X, y])

    search.predict(X_test)
    search.score(X_test, y_test)


@gen_cluster(client=True)
def test_search_plateau_tol(c, s, a, b):
    model = LinearFunction(slope=1)
    params = {"foo": np.linspace(0, 1)}

    # every 3 calls, score will increase by 3. tol=1: model did improved enough
    search = IncrementalSearchCV(model, params, patience=3, tol=1, max_iter=10)
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    yield search.fit(X, y)
    assert set(search.cv_results_["partial_fit_calls"]) == {10}

    # Every 3 calls, score increases by 3. tol=4: model didn't improve enough
    search = IncrementalSearchCV(model, params, patience=3, tol=4, max_iter=10)
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    yield search.fit(X, y)
    assert set(search.cv_results_["partial_fit_calls"]) == {3}


@gen_cluster(client=True)
def test_search_max_iter(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    model = SGDClassifier(tol=1e-3, penalty="elasticnet")
    params = {"alpha": np.logspace(-2, 10, 10), "l1_ratio": np.linspace(0.01, 1, 20)}

    search = IncrementalSearchCV(model, params, n_initial_parameters=10, max_iter=1)
    yield search.fit(X, y, classes=[0, 1])
    for d in search.history_:
        assert d["partial_fit_calls"] <= 1


@gen_cluster(client=True)
@pytest.mark.xfail(
    sys.platform == "win32",
    reason="https://github.com/dask/dask-ml/issues/673",
    strict=False,
)
def test_gridsearch(c, s, a, b):
    def test_gridsearch_func(c, s, a, b):
        X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))

        model = SGDClassifier(tol=1e-3)

        params = {"alpha": np.logspace(-2, 10, 3), "l1_ratio": np.linspace(0.01, 1, 2)}

        search = IncrementalSearchCV(model, params, n_initial_parameters="grid")
        yield search.fit(X, y, classes=[0, 1])

        assert {frozenset(d["params"].items()) for d in search.history_} == {
            frozenset(d.items()) for d in ParameterGrid(params)
        }

    try:
        test_gridsearch_func(c, s, a, b)
    except concurrent.futures.TimeoutError:
        pytest.xfail(reason="https://github.com/dask/dask-ml/issues/673")


@gen_cluster(client=True)
def test_numpy_array(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    X, y = yield c.compute([X, y])
    model = SGDClassifier(tol=1e-3, penalty="elasticnet")
    params = {
        "alpha": np.logspace(-5, -3, 10),
        "l1_ratio": np.linspace(0, 1, 20),
    }

    search = IncrementalSearchCV(model, params, n_initial_parameters=10, max_iter=10)
    yield search.fit(X, y, classes=[0, 1])

    # smoke test to ensure search completed successfully
    assert search.best_score_ > 0


@gen_cluster(client=True)
def test_transform(c, s, a, b):
    def test_transform_func(c, s, a, b):
        X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
        model = MiniBatchKMeans(random_state=0)
        params = {"n_clusters": [3, 4, 5], "n_init": [1, 2]}
        search = IncrementalSearchCV(model, params, n_initial_parameters="grid")
        yield search.fit(X, y)
        (X_,) = yield c.compute([X])
        result = search.transform(X_)
        assert result.shape == (100, search.best_estimator_.n_clusters)

    try:
        test_transform_func(c, s, a, b)
    except concurrent.futures.TimeoutError:
        pytest.xfail(reason="https://github.com/dask/dask-ml/issues/673")


@gen_cluster(client=True)
def test_small(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    model = SGDClassifier(tol=1e-3, penalty="elasticnet")
    params = {"alpha": [0.1, 0.5, 0.75, 1.0]}
    search = IncrementalSearchCV(model, params, n_initial_parameters="grid")
    yield search.fit(X, y, classes=[0, 1])
    (X_,) = yield c.compute([X])
    search.predict(X_)


@gen_cluster(client=True)
def test_smaller(c, s, a, b):
    # infininte loop
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    model = SGDClassifier(tol=1e-3, penalty="elasticnet")
    params = {"alpha": [0.1, 0.5]}
    search = IncrementalSearchCV(model, params, n_initial_parameters="grid")
    yield search.fit(X, y, classes=[0, 1])
    (X_,) = yield c.compute([X])
    search.predict(X_)


def _remove_worst_performing_model(info):
    calls = {v[-1]["partial_fit_calls"] for v in info.values()}
    ests = {v[-1]["params"]["final_score"] for v in info.values()}

    if max(calls) == 1:
        assert all(x in ests for x in [1, 2, 3, 4, 5])
    elif max(calls) == 2:
        assert all(x in ests for x in [2, 3, 4, 5])
        assert all(x not in ests for x in [1])
    elif max(calls) == 3:
        assert all(x in ests for x in [3, 4, 5])
        assert all(x not in ests for x in [1, 2])
    elif max(calls) == 4:
        assert all(x in ests for x in [4, 5])
        assert all(x not in ests for x in [1, 2, 3])
    elif max(calls) == 5:
        assert all(x in ests for x in [5])
        assert all(x not in ests for x in [1, 2, 3, 4])
        return {k: 0 for k in info.keys()}

    recent_scores = {
        k: v[-1]["score"]
        for k, v in info.items()
        if v[-1]["partial_fit_calls"] == max(calls)
    }
    return {k: 1 for k, v in recent_scores.items() if v > min(recent_scores.values())}


@gen_cluster(client=True)
def test_high_performing_models_are_retained_with_patience(c, s, a, b):
    """
    This tests covers a case when high performing models plateau before the
    search is finished.

    This covers the use case when one poor-performing model takes a long time
    to converge, but all other high-performing models have finished (and
    plateaued).

    Details
    -------
    This test defines

    * low performing models that continue to improve
    * high performing models that are constant

    It uses a small tolerance to stop the constant (and high-performing) models.

    This test is only concerned with making sure the high-performing model is
    retained after it has reached a plateau. It is not concerned with making
    sure models are killed off at correct times.
    """

    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))

    params = {"final_score": [1, 2, 3, 4, 5]}
    search = IncrementalSearchCV(
        _MaybeLinearFunction(),
        params,
        patience=2,
        tol=1e-3,  # only stop the constant functions
        n_initial_parameters="grid",
        max_iter=20,
    )

    search._adapt = _remove_worst_performing_model
    yield search.fit(X, y)
    assert search.best_params_ == {"final_score": 5}


@gen_cluster(client=True)
def test_same_params_with_random_state(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=10, chunks=10, random_state=0)
    model = SGDClassifier(tol=1e-3, penalty="elasticnet", random_state=1)
    params = {"alpha": scipy.stats.uniform(1e-4, 1)}

    # Use InverseDecaySearchCV to decay the models and make sure the same ones
    # are selected
    kwargs = dict(n_initial_parameters=10, random_state=2)

    search1 = InverseDecaySearchCV(clone(model), params, **kwargs)
    yield search1.fit(X, y, classes=[0, 1])
    params1 = search1.cv_results_["param_alpha"]

    search2 = InverseDecaySearchCV(clone(model), params, **kwargs)
    yield search2.fit(X, y, classes=[0, 1])
    params2 = search2.cv_results_["param_alpha"]

    assert np.allclose(params1, params2)


@gen_cluster(client=True)
def test_model_random_determinism(c, s, a, b):
    # choose so d == n//10. Then each partial_fit call is very
    # unstable, so models will vary a lot.
    n, d = 50, 5
    X, y = make_classification(
        n_samples=n, n_features=d, chunks=n // 10, random_state=0
    )
    params = {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "average": [True, False],
        "learning_rate": ["constant", "invscaling", "optimal"],
        "eta0": np.logspace(-2, 0, num=1000),
    }

    model = SGDClassifier(random_state=1)
    kwargs = dict(n_initial_parameters=10, random_state=2, max_iter=10)

    search1 = InverseDecaySearchCV(model, params, **kwargs)
    yield search1.fit(X, y, classes=[0, 1])

    search2 = InverseDecaySearchCV(clone(model), params, **kwargs)
    yield search2.fit(X, y, classes=[0, 1])

    assert search1.best_score_ == search2.best_score_
    assert search1.best_params_ == search2.best_params_
    assert np.allclose(search1.best_estimator_.coef_, search2.best_estimator_.coef_)


@gen_cluster(client=True)
def test_min_max_iter(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))
    est = SGDClassifier()
    params = {"alpha": np.logspace(-3, 0)}
    search = IncrementalSearchCV(est, params, max_iter=0)
    with pytest.raises(ValueError, match="max_iter < 1 is not supported"):
        yield search.fit(X, y, classes=[0, 1])


@gen_cluster(client=True)
def test_history(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = IncrementalSearchCV(model, params, max_iter=9, random_state=42)
    yield alg.fit(X, y)
    gt_zero = lambda x: x >= 0
    gt_one = lambda x: x >= 1

    key_types_and_checks = [
        ("mean_partial_fit_time", float, gt_zero),
        ("mean_score_time", float, gt_zero),
        ("std_partial_fit_time", float, gt_zero),
        ("std_score_time", float, gt_zero),
        ("test_score", float, gt_zero),
        ("rank_test_score", int, gt_one),
        ("model_id", int, None),
        ("partial_fit_calls", int, gt_zero),
        ("params", dict, lambda d: set(d.keys()) == {"value"}),
        ("param_value", float, gt_zero),
    ]
    assert set(alg.cv_results_) == {v[0] for v in key_types_and_checks}
    for column, dtype, condition in key_types_and_checks:
        if dtype:
            assert alg.cv_results_[column].dtype == dtype
        if condition:
            assert all(condition(x) for x in alg.cv_results_[column])

    alg.best_estimator_.fit(X, y)
    alg.best_estimator_.score(X, y)
    alg.score(X, y)

    # Test types/format of all parameters we set after fitting
    assert isinstance(alg.best_index_, int)
    assert isinstance(alg.best_estimator_, ConstantFunction)
    assert isinstance(alg.best_score_, float)
    assert isinstance(alg.best_params_, dict)
    assert isinstance(alg.history_, list)
    assert all(isinstance(h, dict) for h in alg.history_)
    assert isinstance(alg.model_history_, dict)
    assert all(vi in alg.history_ for v in alg.model_history_.values() for vi in v)
    assert all(isinstance(v, np.ndarray) for v in alg.cv_results_.values())
    assert isinstance(alg.multimetric_, bool)

    keys = {
        "score",
        "score_time",
        "partial_fit_calls",
        "partial_fit_time",
        "model_id",
        "elapsed_wall_time",
        "params",
    }
    assert all(set(h.keys()) == keys for h in alg.history_)
    times = [v["elapsed_wall_time"] for v in alg.history_]
    assert (np.diff(times) >= 0).all()

    # Test to make sure history_ ordered with wall time
    assert (np.diff([v["elapsed_wall_time"] for v in alg.history_]) >= 0).all()
    for model_hist in alg.model_history_.values():
        calls = [h["partial_fit_calls"] for h in model_hist]
        assert (np.diff(calls) >= 1).all() or len(calls) == 1


@pytest.mark.parametrize("Search", [HyperbandSearchCV, IncrementalSearchCV])
@pytest.mark.parametrize("verbose", [True, False])
def test_verbosity(Search, verbose, capsys):
    max_iter = 15

    @gen_cluster(client=True)
    async def _test_verbosity(c, s, a, b):
        X, y = make_classification(n_samples=10, n_features=4, chunks=10)
        model = ConstantFunction()
        params = {"value": scipy.stats.uniform(0, 1)}
        search = Search(model, params, max_iter=max_iter, verbose=verbose)
        await search.fit(X, y)
        assert search.best_score_ > 0  # ensure search ran
        return search

    # IncrementalSearchCV always logs to INFO
    logger = logging.getLogger("dask_ml.model_selection")
    with captured_logger(logger) as logs:
        _test_verbosity()
        messages = logs.getvalue().splitlines()

    # Make sure we always log
    assert messages
    assert any("score" in m for m in messages)

    # If verbose=True, make sure logs to stdout
    _test_verbosity()
    std = capsys.readouterr()
    stdout = [line for line in std.out.split("\n") if line]
    if verbose:
        assert len(stdout) >= 1
        assert all(["CV" in line for line in stdout])
    else:
        assert not len(stdout)

    if "Hyperband" in str(Search):
        assert all("[CV, bracket=" in m for m in messages)
    else:
        assert all("[CV]" in m for m in messages)

    brackets = 3 if "Hyperband" in str(Search) else 1
    assert sum("examples in each chunk" in m for m in messages) == brackets
    assert sum("creating" in m and "models" in m for m in messages) == brackets


@gen_cluster(client=True)
def test_verbosity_types(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}

    for verbose in [-1.0, 1.2]:
        search = IncrementalSearchCV(model, params, verbose=verbose, max_iter=3)
        with pytest.raises(ValueError, match="0 <= verbose <= 1"):
            yield search.fit(X, y)

    for verbose in [0.0, 0, 1, 1.0, True, False]:
        search = IncrementalSearchCV(model, params, verbose=verbose, max_iter=3)
        yield search.fit(X, y)


@pytest.mark.parametrize("verbose", [0, 0.0, 1 / 2, 1, 1.0, False, True])
def test_verbosity_levels(capsys, verbose):
    max_iter = 14

    @gen_cluster(client=True)
    def _test_verbosity(c, s, a, b):
        X, y = make_classification(n_samples=10, n_features=4, chunks=10)
        model = ConstantFunction()
        params = {"value": scipy.stats.uniform(0, 1)}
        search = IncrementalSearchCV(model, params, max_iter=max_iter, verbose=verbose)
        yield search.fit(X, y)
        return search

    with captured_logger(logging.getLogger("dask_ml.model_selection")) as logs:
        search = _test_verbosity()
        assert search.best_score_ > 0  # ensure search ran
        messages = logs.getvalue().splitlines()

    factor = 1 if isinstance(verbose, bool) else verbose
    assert len(messages) == pytest.approx(max_iter * factor + 2, abs=1)


@gen_cluster(client=True)
def test_search_patience_infeasible_tol(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))

    rng = check_random_state(42)
    params = {"value": rng.rand(1000)}
    model = ConstantFunction()

    max_iter = 10
    score_increase = -10
    search = IncrementalSearchCV(
        model, params, max_iter=max_iter, patience=3, tol=score_increase,
    )
    yield search.fit(X, y, classes=[0, 1])

    hist = pd.DataFrame(search.history_)
    assert hist.partial_fit_calls.max() == max_iter


@gen_cluster(client=True)
def test_search_basic_patience(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=(10, 5))

    rng = check_random_state(42)
    params = {"slope": 2 + rng.rand(1000)}
    model = LinearFunction()

    # Test the case where tol to small (all models finish)
    max_iter = 15
    patience = 5
    increase_after_patience = patience
    search = IncrementalSearchCV(
        model,
        params,
        max_iter=max_iter,
        tol=increase_after_patience,
        patience=patience,
        fits_per_score=3,
    )
    yield search.fit(X, y, classes=[0, 1])

    hist = pd.DataFrame(search.history_)
    # +1 (and +2 below) because scores_per_fit isn't exact
    assert hist.partial_fit_calls.max() == max_iter + 1

    # Test the case where tol to large (no models finish)
    patience = 5
    increase_after_patience = patience
    params = {"slope": 0 + 0.9 * rng.rand(1000)}
    search = IncrementalSearchCV(
        model,
        params,
        max_iter=max_iter,
        tol=increase_after_patience,
        patience=patience,
        fits_per_score=3,
    )
    yield search.fit(X, y, classes=[0, 1])

    hist = pd.DataFrame(search.history_)
    assert hist.partial_fit_calls.max() == patience + 2


@gen_cluster(client=True)
def test_search_invalid_patience(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=10)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()

    search = IncrementalSearchCV(model, params, patience=1, max_iter=10)
    with pytest.raises(ValueError, match="patience >= 2"):
        yield search.fit(X, y, classes=[0, 1])

    search = IncrementalSearchCV(model, params, patience=2.0, max_iter=10)
    with pytest.raises(ValueError, match="patience must be an integer"):
        yield search.fit(X, y, classes=[0, 1])

    # Make sure this passes
    search = IncrementalSearchCV(model, params, patience=False, max_iter=10)
    yield search.fit(X, y, classes=[0, 1])
    assert search.history_


@gen_cluster(client=True)
def test_warns_scores_per_fit(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=10)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()

    search = IncrementalSearchCV(model, params, scores_per_fit=2)
    with pytest.warns(UserWarning, match="deprecated since Dask-ML v1.4.0"):
        yield search.fit(X, y)


@gen_cluster(client=True)
async def test_model_future(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=10)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()
    model_future = await c.scatter(model)

    search = IncrementalSearchCV(model_future, params, max_iter=10)

    await search.fit(X, y, classes=[0, 1])
    assert search.history_
    assert search.best_score_ > 0
