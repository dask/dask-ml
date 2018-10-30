import dask.array as da
import numpy as np
import pytest
import scipy.stats
import sklearn.datasets
from dask.distributed import Client
from distributed.utils_test import cluster, gen_cluster, loop  # noqa: F401
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterSampler
from toolz import partial
from tornado import gen

from dask_ml.datasets import make_classification
from dask_ml.model_selection import HyperbandCV
from dask_ml.model_selection._incremental import fit as incremental_fit
from dask_ml.utils import ConstantFunction
from dask_ml.wrappers import Incremental
from dask_ml.model_selection._successive_halving import SuccessiveHalving


@gen_cluster(client=True, timeout=5000)
def test_basic(c, s, a, b):
    for array_type, library, max_iter in [
        ("dask.array", "dask-ml", 9),
        ("numpy", "sklearn", 9),
        ("numpy", "ConstantFunction", 27),
        ("numpy", "ConstantFunction", 81),
    ]:
        print(" " * 20, array_type, library, max_iter)
        yield _test_basic(array_type, library, max_iter, c)


@gen.coroutine
def _test_basic(array_type, library, max_iter, c):
    n, d = (200, 2)

    rng = da.random.RandomState(42)

    # create observations we know linear models can fit
    X = rng.normal(size=(n, d), chunks=n // 2)
    coef_star = rng.uniform(size=d, chunks=d)
    y = da.sign(X.dot(coef_star))

    if array_type == "numpy":
        X = yield c.compute(X)
        y = yield c.compute(y)

    params = {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "average": [True, False],
        "learning_rate": ["constant", "invscaling", "optimal"],
        "eta0": np.logspace(-2, 0, num=1000),
    }
    model = SGDClassifier(tol=-np.inf, penalty="elasticnet", random_state=42, eta0=0.1)
    if library == "dask-ml":
        model = Incremental(model)
        params = {"estimator__" + k: v for k, v in params.items()}
    elif library == "ConstantFunction":
        model = ConstantFunction()
        params = {"value": np.linspace(0, 1, num=1000)}

    search = HyperbandCV(model, params, max_iter=max_iter, random_state=42)
    classes = c.compute(da.unique(y))
    yield search.fit(X, y, classes=classes)

    X, y = sklearn.datasets.make_classification(
        n_features=d, n_informative=d, n_repeated=0, n_redundant=0
    )
    score = search.best_estimator_.score(X, y)
    assert type(search.best_estimator_) == type(model)
    assert isinstance(search.best_params_, dict)

    num_fit_models = len(set(search.cv_results_["model_id"]))
    num_models = {9: 17, 27: 49, 81: 143}
    assert num_fit_models == num_models[max_iter]
    best_idx = search.best_index_
    if isinstance(model, ConstantFunction):
        assert search.cv_results_["test_score"][best_idx] == max(
            search.cv_results_["test_score"]
        )
    model_ids = {h["model_id"] for h in search.history_}
    assert len(model_ids) > max_iter
    assert all("bracket" in id_ for id_ in model_ids)


@pytest.mark.parametrize("max_iter,aggressiveness", [(9, 3), (27, 3), (64, 4)])
def test_hyperband_mirrors_paper(loop, max_iter, aggressiveness):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):

            X, y = make_classification(chunks=5, n_features=5)
            model = ConstantFunction()
            params = {"value": np.random.rand(max_iter)}
            alg = HyperbandCV(
                model,
                params,
                max_iter=max_iter,
                random_state=0,
                aggressiveness=aggressiveness,
            )
            alg.fit(X, y)
            metadata = alg.metadata()
            paper_iters = [b.pop("iters") for b in metadata["brackets"].values()]
            actual_iters = [b.pop("iters") for b in alg.metadata_["brackets"].values()]
            assert metadata == alg.metadata_
            for paper_iter, actual_iter in zip(paper_iters, actual_iters):
                assert set(paper_iter).issubset(set(actual_iter))
            assert alg.best_score_ == params["value"].max()


def test_hyperband_patience(loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):

            X, y = make_classification(chunks=5, n_features=5)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(
                model, params, max_iter=27, random_state=0, patience=10, tol=1e-3
            )

            alg.fit(X, y)

            actual_iters = [b.pop("iters") for b in alg.metadata_["brackets"].values()]
            paper_iters = [b.pop("iters") for b in alg.metadata()["brackets"].values()]
            for paper_iter, actual_iter in zip(paper_iters, actual_iters):
                paper_iter = {k for k in paper_iter if k <= 15}
                assert set(paper_iter).issubset(actual_iter)
                assert all(x <= 15 for x in actual_iter)
            assert (
                alg.metadata_["partial_fit_calls"]
                <= alg.metadata()["partial_fit_calls"]
            )
            assert alg.metadata_["models"] == alg.metadata()["models"]


def test_integration(loop):  # noqa: F811
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, max_iter=9, random_state=42)
            alg.fit(X, y)
            cv_res_keys = set(alg.cv_results_.keys())
            gt_zero = lambda x: x >= 0
            for column, dtype, condition in [
                ("params", dict, lambda d: set(d.keys()) == {"value"}),
                ("test_score", float, None),
                ("test_score", float, None),
                ("rank_test_score", int, None),
                ("mean_partial_fit_time", float, gt_zero),
                ("std_partial_fit_time", float, gt_zero),
                ("mean_score_time", float, gt_zero),
                ("std_score_time", float, gt_zero),
                ("model_id", str, None),
                ("partial_fit_calls", int, gt_zero),
                ("param_value", float, None),
            ]:
                if dtype:
                    assert all(isinstance(x, dtype) for x in alg.cv_results_[column])
                if condition:
                    assert all(condition(x) for x in alg.cv_results_[column])
                cv_res_keys -= {column}

            # the keys listed in the for-loop are all the keys in cv_results_
            assert cv_res_keys == set()

            alg.best_estimator_.fit(X, y)
            alg.best_estimator_.score(X, y)
            alg.fit(X, y)
            alg.score(X, y)
            assert isinstance(alg.best_index_, int)
            assert isinstance(alg.best_score_, float)
            assert isinstance(alg.best_estimator_, ConstantFunction)
            assert isinstance(alg.best_params_, dict)
            assert isinstance(alg.history_, list)
            assert all("bracket=" in h["model_id"] for h in alg.history_)
            keys = {
                "score",
                "score_time",
                "partial_fit_calls",
                "partial_fit_time",
                "model_id",
                "bracket",
                "wall_time",
                "params",
            }
            assert all(set(h.keys()) == keys for h in alg.history_)


@gen_cluster(client=True, timeout=5000)
def test_successive_halving_params(c, s, a, b):
    X, y = make_classification(n_samples=10, n_features=4, chunks=10)
    model = ConstantFunction()
    params = {"value": scipy.stats.uniform(0, 1)}
    alg = HyperbandCV(model, params, max_iter=9, random_state=42)

    kwargs = {
        k: v["SuccessiveHalving params"] for k, v in alg.metadata()["brackets"].items()
    }
    SHAs = {k: SuccessiveHalving(model, params, **v) for k, v in kwargs.items()}

    metadata = alg.metadata()["brackets"]
    for b, SHA in SHAs.items():
        yield SHA.fit(X, y)
        assert metadata[b]["models"] == SHA.metadata_["models"]
        assert metadata[b]["partial_fit_calls"] == SHA.metadata_["partial_fit_calls"]
