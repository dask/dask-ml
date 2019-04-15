import math
from collections import defaultdict

import dask.array as da
import numpy as np
import pytest
import scipy.stats
from distributed.utils_test import cluster, gen_cluster, loop  # noqa: F401
from sklearn.linear_model import SGDClassifier

from dask_ml.datasets import make_classification
from dask_ml.model_selection import (
    HyperbandSearchCV,
    IncrementalSearchCV,
    SuccessiveHalvingSearchCV,
)
from dask_ml.model_selection._hyperband import _get_hyperband_params
from dask_ml.utils import ConstantFunction
from dask_ml.wrappers import Incremental


@pytest.mark.parametrize(
    "array_type, library, max_iter",
    [
        ("dask.array", "dask-ml", 9),
        ("numpy", "sklearn", 9),
        ("numpy", "ConstantFunction", 15),
        ("numpy", "ConstantFunction", 20),
    ],
)
def test_basic(array_type, library, max_iter):
    @gen_cluster(client=True, timeout=5000)
    def _test_basic(c, s, a, b):
        n, d = (50, 2)

        rng = da.random.RandomState(42)

        # create observations we know linear models can fit
        X = rng.normal(size=(n, d), chunks=n // 2)
        coef_star = rng.uniform(size=d, chunks=d)
        y = da.sign(X.dot(coef_star))

        if array_type == "numpy":
            X, y = yield c.compute((X, y))

        params = {
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            "average": [True, False],
            "learning_rate": ["constant", "invscaling", "optimal"],
            "eta0": np.logspace(-2, 0, num=1000),
        }
        model = SGDClassifier(
            tol=-np.inf, penalty="elasticnet", random_state=42, eta0=0.1
        )
        if library == "dask-ml":
            model = Incremental(model)
            params = {"estimator__" + k: v for k, v in params.items()}
        elif library == "ConstantFunction":
            model = ConstantFunction()
            params = {"value": np.linspace(0, 1, num=1000)}

        search = HyperbandSearchCV(model, params, max_iter=max_iter, random_state=42)
        classes = c.compute(da.unique(y))
        yield search.fit(X, y, classes=classes)

        if library == "dask-ml":
            X, y = yield c.compute((X, y))
        score = search.best_estimator_.score(X, y)
        assert score == search.score(X, y)
        assert 0 <= score <= 1

        if library == "ConstantFunction":
            assert score == search.best_score_
        else:
            assert abs(score - search.best_score_) < 0.1

        assert type(search.best_estimator_) == type(model)
        assert isinstance(search.best_params_, dict)

        num_fit_models = len(set(search.cv_results_["model_id"]))
        num_pf_calls = sum(
            [v[-1]["partial_fit_calls"] for v in search.model_history_.values()]
        )
        models = {9: 17, 15: 17, 20: 17, 27: 49, 30: 49, 81: 143}
        pf_calls = {9: 69, 15: 101, 20: 144, 27: 357, 30: 379, 81: 1581}
        assert num_fit_models == models[max_iter]
        assert num_pf_calls == pf_calls[max_iter]

        best_idx = search.best_index_
        if isinstance(model, ConstantFunction):
            assert search.cv_results_["test_score"][best_idx] == max(
                search.cv_results_["test_score"]
            )
        model_ids = {h["model_id"] for h in search.history_}

        if math.log(max_iter, 3) % 1.0 == 0:
            # log(max_iter, 3) % 1.0 == 0 is the good case when max_iter is a
            # power of search.aggressiveness
            # In this case, assert that more models are tried then the max_iter
            assert len(model_ids) > max_iter
        else:
            # Otherwise, give some padding "almost as many estimators are tried
            # as max_iter". 3 is a fudge number chosen to be the minimum; when
            # max_iter=20, len(model_ids) == 17.
            assert len(model_ids) + 3 >= max_iter
        assert all("bracket" in id_ for id_ in model_ids)

    _test_basic()


@pytest.mark.parametrize("max_iter,aggressiveness", [(27, 3), (30, 4)])
def test_hyperband_mirrors_paper(max_iter, aggressiveness):
    @gen_cluster(client=True, timeout=5000)
    def _test_mirrors_paper(c, s, a, b):
        X, y = make_classification(n_samples=10, n_features=4, chunks=10)
        model = ConstantFunction()
        params = {"value": np.random.rand(max_iter)}
        alg = HyperbandSearchCV(
            model,
            params,
            max_iter=max_iter,
            random_state=0,
            aggressiveness=aggressiveness,
        )
        yield alg.fit(X, y)
        metadata = alg.metadata()
        paper_decisions = [b.pop("decisions") for b in metadata["brackets"].values()]
        actual_decisions = [
            b.pop("decisions") for b in alg.metadata_["brackets"].values()
        ]
        assert metadata == alg.metadata_
        for paper_iter, actual_iter in zip(paper_decisions, actual_decisions):
            assert set(paper_iter).issubset(set(actual_iter))
        if aggressiveness == 3:
            assert alg.best_score_ == params["value"].max()

    _test_mirrors_paper()


@gen_cluster(client=True, timeout=5000)
def test_hyperband_patience(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    max_iter = 27

    alg = HyperbandSearchCV(
        model, params, max_iter=max_iter, patience=True, random_state=0
    )
    yield alg.fit(X, y)

    alg_patience = max_iter // alg.aggressiveness
    actual_decisions = [b.pop("decisions") for b in alg.metadata_["brackets"].values()]
    paper_decisions = [b.pop("decisions") for b in alg.metadata()["brackets"].values()]

    for paper_iter, actual_iter in zip(paper_decisions, actual_decisions):
        trimmed_paper_iter = {k for k in paper_iter if k <= alg_patience}
        assert trimmed_paper_iter.issubset(set(actual_iter))
        assert all(x <= alg_patience + 1 for x in actual_iter)

    assert alg.metadata_["partial_fit_calls"] <= alg.metadata()["partial_fit_calls"]
    assert alg.best_score_ >= 0.9

    alg = HyperbandSearchCV(model, params, max_iter=max_iter, patience=1)
    with pytest.warns(UserWarning, match="The goal of `patience`"):
        yield alg.fit(X, y)


@gen_cluster(client=True, timeout=5000)
def test_integration(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=9, random_state=42)
    yield alg.fit(X, y)
    cv_res_keys = set(alg.cv_results_.keys())
    gt_zero = lambda x: x >= 0
    gt_one = lambda x: x >= 1
    for column, dtype, condition in [
        ("params", dict, lambda d: set(d.keys()) == {"value"}),
        ("test_score", float, gt_zero),
        ("test_score", float, gt_zero),
        ("rank_test_score", int, gt_one),
        ("mean_partial_fit_time", float, gt_zero),
        ("std_partial_fit_time", float, gt_zero),
        ("mean_score_time", float, gt_zero),
        ("std_score_time", float, gt_zero),
        ("model_id", None, lambda x: isinstance(x, str)),
        ("partial_fit_calls", int, gt_zero),
        ("param_value", float, gt_zero),
    ]:
        if dtype:
            assert alg.cv_results_[column].dtype == dtype
        if condition:
            assert all(condition(x) for x in alg.cv_results_[column])
        cv_res_keys -= {column}

    # the keys listed in the for-loop are all the keys in cv_results_
    assert cv_res_keys == set()

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
    assert all("bracket=" in h["model_id"] for h in alg.history_)

    keys = {
        "score",
        "score_time",
        "partial_fit_calls",
        "partial_fit_time",
        "model_id",
        "bracket",
        "elapsed_wall_time",
        "params",
    }
    assert all(set(h.keys()) == keys for h in alg.history_)
    times = [v["elapsed_wall_time"] for v in alg.history_]
    assert (np.diff(times) >= 0).all()
    history = defaultdict(list)
    for h in alg.history_:
        history[h["model_id"]] += [h]
    for model_hist in history.values():
        calls = [h["partial_fit_calls"] for h in model_hist]
        assert (np.diff(calls) >= 1).all() or len(calls) == 1


@gen_cluster(client=True, timeout=5000)
def test_successive_halving_params(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=9, random_state=42)

    kwargs = {
        k: v["SuccessiveHalvingSearchCV params"]
        for k, v in alg.metadata()["brackets"].items()
    }
    SHAs = {k: SuccessiveHalvingSearchCV(**v) for k, v in kwargs.items()}

    metadata = alg.metadata()["brackets"]
    for b, SHA in SHAs.items():
        yield SHA.fit(X, y)
        estimators = len(SHA.model_history_)
        pf_calls = [v[-1]["partial_fit_calls"] for v in SHA.model_history_.values()]
        assert metadata[b]["estimators"] == estimators
        assert metadata[b]["partial_fit_calls"] == sum(pf_calls)


def test_correct_params():
    est = ConstantFunction()
    params = {"value": [0, 1]}
    search = HyperbandSearchCV(est, params)

    base = {
        "estimator",
        "estimator__value",
        "estimator__sleep",
        "parameters",
        "max_iter",
        "test_size",
        "patience",
        "tol",
        "random_state",
        "scoring",
    }
    assert set(search.get_params().keys()) == base.union({"aggressiveness"})
    meta = search.metadata()
    SHAs_params = [
        bracket["SuccessiveHalvingSearchCV params"]
        for bracket in meta["brackets"].values()
    ]
    SHA_params = base.union(
        {
            "n_initial_parameters",
            "n_initial_iter",
            "aggressiveness",
            "max_iter",
        }
    ) - {"estimator__value", "estimator__sleep"}
    assert all(set(SHA) == SHA_params for SHA in SHAs_params)


def test_params_passed():
    params = {
        "aggressiveness": 3.5,
        "estimator": ConstantFunction(value=0.4),
        "max_iter": 253,
        "random_state": 42,
        "scoring": False,
        "test_size": 0.212,
        "tol": 0,
        "parameters": {"value": [0, 1]},
    }
    params["patience"] = (params["max_iter"] // params["aggressiveness"]) + 4
    hyperband = HyperbandSearchCV(**params)

    for k, v in params.items():
        assert getattr(hyperband, k) == v

    brackets = hyperband.metadata()["brackets"]
    SHAs_params = [
        bracket["SuccessiveHalvingSearchCV params"] for bracket in brackets.values()
    ]

    for SHA_params in SHAs_params:
        for k, v in params.items():
            if k == "random_state":
                continue
            assert SHA_params[k] == v
    seeds = [SHA_params["random_state"] for SHA_params in SHAs_params]
    assert len(set(seeds)) == len(seeds)


@gen_cluster(client=True, timeout=5000)
def test_same_random_state_same_params(c, s, a, b):
    seed = 0
    values = scipy.stats.uniform(0, 1)
    h = HyperbandSearchCV(
        ConstantFunction(), {"value": values}, random_state=seed, max_iter=9
    )

    # Make a class for passive random sampling
    passive = IncrementalSearchCV(
        ConstantFunction(),
        {"value": values},
        random_state=seed,
        max_iter=2,
        n_initial_parameters=h.metadata()["estimators"],
    )
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    yield h.fit(X, y)
    yield passive.fit(X, y)

    # Check to make sure the Hyperbands found the same params
    v_h = h.cv_results_["param_value"]

    # Check to make sure the random passive search had *some* of the same params
    v_passive = passive.cv_results_["param_value"]
    # Sanity checks to make sure all unique floats
    assert len(set(v_passive)) == len(v_passive)
    assert len(set(v_h)) == len(v_h)

    # Getting the `value`s that are the same for both searches
    same = set(v_passive).intersection(set(v_h))

    passive_models = h.metadata()["brackets"]["bracket=0"]["estimators"]
    assert len(same) == passive_models


def test_random_state_no_seed_different_params():
    values = scipy.stats.uniform(0, 1)
    max_iter = 9
    brackets = _get_hyperband_params(max_iter)

    h1 = HyperbandSearchCV(ConstantFunction(), {"value": values}, max_iter=max_iter)
    h2 = HyperbandSearchCV(ConstantFunction(), {"value": values}, max_iter=max_iter)

    h1._get_SHAs(brackets)
    h2._get_SHAs(brackets)

    assert h1._SHA_seed != h2._SHA_seed

    h1 = HyperbandSearchCV(
        ConstantFunction(), {"value": values}, max_iter=9, random_state=0
    )
    h2 = HyperbandSearchCV(
        ConstantFunction(), {"value": values}, max_iter=9, random_state=0
    )

    h1._get_SHAs(brackets)
    h2._get_SHAs(brackets)

    assert h1._SHA_seed == h2._SHA_seed
