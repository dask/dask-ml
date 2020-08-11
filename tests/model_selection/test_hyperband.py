import logging
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from dask import array as da, dataframe as dd
from distributed.utils_test import (  # noqa: F401
    captured_logger,
    cluster,
    gen_cluster,
    loop,
)
from sklearn.linear_model import SGDClassifier

from dask_ml._compat import DISTRIBUTED_2_5_0
from dask_ml.datasets import make_classification
from dask_ml.model_selection import (
    HyperbandSearchCV,
    IncrementalSearchCV,
    SuccessiveHalvingSearchCV,
)
from dask_ml.model_selection._hyperband import _get_hyperband_params
from dask_ml.utils import ConstantFunction
from dask_ml.wrappers import Incremental

pytestmark = pytest.mark.skipif(not DISTRIBUTED_2_5_0, reason="hangs")


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
    @gen_cluster(client=True)
    def _test_basic(c, s, a, b):
        rng = da.random.RandomState(42)

        n, d = (50, 2)
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
            # These are not equal because IncrementalSearchCV uses a train/test
            # split and we're testing on the entire train dataset, not only the
            # validation/test set.
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
def test_hyperband_mirrors_paper_and_metadata(max_iter, aggressiveness):
    @gen_cluster(client=True)
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

        assert alg.metadata == alg.metadata_

        assert isinstance(alg.metadata["brackets"], list)
        assert set(alg.metadata.keys()) == {"n_models", "partial_fit_calls", "brackets"}

        # Looping over alg.metadata["bracketes"] is okay because alg.metadata
        # == alg.metadata_
        for bracket in alg.metadata["brackets"]:
            assert set(bracket.keys()) == {
                "n_models",
                "partial_fit_calls",
                "bracket",
                "SuccessiveHalvingSearchCV params",
                "decisions",
            }

        if aggressiveness == 3:
            assert alg.best_score_ == params["value"].max()

    _test_mirrors_paper()


@gen_cluster(client=True)
def test_hyperband_patience(c, s, a, b):
    # Test to make sure that specifying patience=True results in less
    # computation
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    max_iter = 27

    alg = HyperbandSearchCV(
        model, params, max_iter=max_iter, patience=True, random_state=0
    )
    yield alg.fit(X, y)

    alg_patience = max_iter // alg.aggressiveness
    actual_decisions = [b.pop("decisions") for b in alg.metadata_["brackets"]]
    paper_decisions = [b.pop("decisions") for b in alg.metadata["brackets"]]

    for paper_iter, actual_iter in zip(paper_decisions, actual_decisions):
        trimmed_paper_iter = {k for k in paper_iter if k <= alg_patience}

        # This makes sure that the algorithm is executed faithfully when
        # patience=True (and the proper decision points are preserved even if
        # other stop-on-plateau points are added)
        assert trimmed_paper_iter.issubset(set(actual_iter))

        # This makes sure models aren't trained for too long
        assert all(x <= alg_patience + 1 for x in actual_iter)

    assert alg.metadata_["partial_fit_calls"] <= alg.metadata["partial_fit_calls"]
    assert alg.best_score_ >= 0.9

    max_iter = 6
    kwargs = dict(max_iter=max_iter, aggressiveness=2)
    alg = HyperbandSearchCV(model, params, patience=2, **kwargs)
    with pytest.warns(UserWarning, match="The goal of `patience`"):
        yield alg.fit(X, y)

    alg = HyperbandSearchCV(model, params, patience=2, tol=np.nan, **kwargs)
    yield alg.fit(X, y)
    assert pd.DataFrame(alg.history_).partial_fit_calls.max() == max_iter

    alg = HyperbandSearchCV(model, params, patience=2, tol=None, **kwargs)
    yield alg.fit(X, y)
    assert pd.DataFrame(alg.history_).partial_fit_calls.max() == max_iter

    alg = HyperbandSearchCV(model, params, patience=1, **kwargs)
    with pytest.raises(ValueError, match="always detect a plateau"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield alg.fit(X, y)


@gen_cluster(client=True)
def test_cv_results_order_preserved(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=9, random_state=42)
    yield alg.fit(X, y)

    info = {k: v[-1] for k, v in alg.model_history_.items()}
    for _, row in pd.DataFrame(alg.cv_results_).iterrows():
        model_info = info[row["model_id"]]
        assert row["bracket"] == model_info["bracket"]
        assert row["params"] == model_info["params"]
        assert np.allclose(row["test_score"], model_info["score"])


@gen_cluster(client=True)
def test_successive_halving_params(c, s, a, b):
    # Makes sure when SHAs are fit with values from the "SuccessiveHalvingSearchCV
    # params" key, the number of models/calls stay the same as Hyperband.

    # This sanity check again makes sure parameters passed correctly
    # (similar to `test_params_passed`)
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=27, random_state=42)

    kwargs = [v["SuccessiveHalvingSearchCV params"] for v in alg.metadata["brackets"]]
    SHAs = [SuccessiveHalvingSearchCV(model, params, **v) for v in kwargs]

    metadata = alg.metadata["brackets"]
    for k, (true_meta, SHA) in enumerate(zip(metadata, SHAs)):
        yield SHA.fit(X, y)
        n_models = len(SHA.model_history_)
        pf_calls = [v[-1]["partial_fit_calls"] for v in SHA.model_history_.values()]
        assert true_meta["n_models"] == n_models
        assert true_meta["partial_fit_calls"] == sum(pf_calls)


@gen_cluster(client=True)
def test_correct_params(c, s, a, b):
    # Makes sure that Hyperband has the correct parameters.

    # Implemented because Hyperband wraps SHA. Again, makes sure that parameters
    # are correctly passed to SHA (had a case where max_iter= flag not passed to
    # SuccessiveHalvingSearchCV but it should have been)
    est = ConstantFunction()
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    params = {"value": np.linspace(0, 1)}
    search = HyperbandSearchCV(est, params, max_iter=9)

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
        "verbose",
        "prefix",
    }
    assert set(search.get_params().keys()) == base.union({"aggressiveness"})
    meta = search.metadata
    SHAs_params = [
        bracket["SuccessiveHalvingSearchCV params"] for bracket in meta["brackets"]
    ]
    SHA_params = base.union(
        {
            "n_initial_parameters",
            "n_initial_iter",
            "aggressiveness",
            "max_iter",
            "prefix",
        }
    ) - {"estimator__sleep", "estimator__value", "estimator", "parameters"}

    assert all(set(SHA) == SHA_params for SHA in SHAs_params)

    # this is testing to make sure that each SHA has the correct estimator
    yield search.fit(X, y)
    SHAs = search._SuccessiveHalvings_
    assert all(search.estimator is SHA.estimator for SHA in SHAs.values())
    assert all(search.parameters is SHA.parameters for SHA in SHAs.values())


def test_params_passed():
    # This makes sure that the "SuccessiveHalvingSearchCV params" key in
    # Hyperband.metadata["brackets"] is correct and they can be instatiated.
    est = ConstantFunction(value=0.4)
    params = {"value": np.linspace(0, 1)}
    params = {
        "aggressiveness": 3.5,
        "max_iter": 253,
        "random_state": 42,
        "scoring": False,
        "test_size": 0.212,
        "tol": 0,
    }
    params["patience"] = (params["max_iter"] // params["aggressiveness"]) + 4
    hyperband = HyperbandSearchCV(est, params, **params)

    for k, v in params.items():
        assert getattr(hyperband, k) == v

    brackets = hyperband.metadata["brackets"]
    SHAs_params = [bracket["SuccessiveHalvingSearchCV params"] for bracket in brackets]

    for SHA_params in SHAs_params:
        for k, v in params.items():
            if k == "random_state":
                continue
            assert SHA_params[k] == v
    seeds = [SHA_params["random_state"] for SHA_params in SHAs_params]
    assert len(set(seeds)) == len(seeds)


# decay_rate warnings are tested in test_incremental_warns.py
@pytest.mark.filterwarnings("ignore:decay_rate")
@gen_cluster(client=True)
def test_same_random_state_same_params(c, s, a, b):
    # This makes sure parameters are sampled correctly when random state is
    # specified.

    # This test makes sure random state is *correctly* passed to successive
    # halvings from Hyperband
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
        n_initial_parameters=h.metadata["n_models"],
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

    passive_models = h.metadata["brackets"][0]["n_models"]
    assert len(same) == passive_models


def test_random_state_no_seed_different_params():
    # Guarantees that specifying a random state relies in the successive
    # halving's having the same random state (and likewise for no random state)

    # This test is required because Hyperband wraps SHAs and the random state
    # needs to be passed correctly.
    values = scipy.stats.uniform(0, 1)
    max_iter = 9
    brackets = _get_hyperband_params(max_iter)
    kwargs = {"value": values}

    h1 = HyperbandSearchCV(ConstantFunction(), kwargs, max_iter=max_iter)
    h2 = HyperbandSearchCV(ConstantFunction(), kwargs, max_iter=max_iter)
    h1._get_SHAs(brackets)
    h2._get_SHAs(brackets)
    assert h1._SHA_seed != h2._SHA_seed

    h1 = HyperbandSearchCV(ConstantFunction(), kwargs, max_iter=9, random_state=0)
    h2 = HyperbandSearchCV(ConstantFunction(), kwargs, max_iter=9, random_state=0)
    h1._get_SHAs(brackets)
    h2._get_SHAs(brackets)
    assert h1._SHA_seed == h2._SHA_seed


@gen_cluster(client=True)
def test_min_max_iter(c, s, a, b):
    # This test makes sure Hyperband works with max_iter=1.
    # Tests for max_iter < 1 are in test_incremental.py.
    values = scipy.stats.uniform(0, 1)
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)

    max_iter = 1
    h = HyperbandSearchCV(ConstantFunction(), {"value": values}, max_iter=max_iter)
    yield h.fit(X, y)
    assert h.best_score_ > 0


@gen_cluster(client=True)
def test_history(c, s, a, b):
    # This test is required to make sure Hyperband wraps SHA successfully
    # Mostly, it's a test to make sure ordered by time
    #
    # There's also a test in test_incremental to make sure the history has
    # correct values/etc
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=9, random_state=42)
    yield alg.fit(X, y)

    assert alg.cv_results_["model_id"].dtype == "<U11"
    assert all(isinstance(v, str) for v in alg.cv_results_["model_id"])
    assert all("bracket=" in h["model_id"] for h in alg.history_)
    assert all("bracket" in h for h in alg.history_)

    # Hyperband does a custom ordering of times
    times = [v["elapsed_wall_time"] for v in alg.history_]
    assert (np.diff(times) >= 0).all()
    # Make sure results are ordered by partial fit calls for each model
    for model_hist in alg.model_history_.values():
        calls = [h["partial_fit_calls"] for h in model_hist]
        assert (np.diff(calls) >= 1).all() or len(calls) == 1


@gen_cluster(client=True)
def test_logs_dont_repeat(c, s, a, b):
    # This test is necessary to make sure the dask_ml.model_selection logger
    # isn't piped to stdout repeatedly.
    #
    # I developed this test to protect against this case:
    # getLogger("dask_ml.model_selection") is piped to stdout whenever a
    # bracket of Hyperband starts/each time SHA._fit is called
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    search = HyperbandSearchCV(model, params, max_iter=9, random_state=42)
    with captured_logger(logging.getLogger("dask_ml.model_selection")) as logs:
        yield search.fit(X, y)
        assert search.best_score_ > 0  # ensure search ran
        messages = logs.getvalue().splitlines()
    model_creation_msgs = [m for m in messages if "creating" in m]
    n_models = [m.split(" ")[-2] for m in model_creation_msgs]

    bracket_models = [b["n_models"] for b in search.metadata["brackets"]]
    assert len(bracket_models) == len(set(bracket_models))

    # Make sure only one model creation message is printed per bracket
    # (all brackets have unique n_models as asserted above)
    assert len(n_models) == len(set(n_models))


@gen_cluster(client=True)
async def test_dataframe_inputs(c, s, a, b):
    X = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    X = dd.from_pandas(X, npartitions=2)
    y = pd.Series([False, True, True])
    y = dd.from_pandas(y, npartitions=2)

    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandSearchCV(model, params, max_iter=9, random_state=42)
    await alg.fit(X, y)
