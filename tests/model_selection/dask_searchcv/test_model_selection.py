from __future__ import absolute_import, division, print_function

import os
import pickle
from itertools import product
from multiprocessing import cpu_count

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.base import tokenize
from dask.callbacks import Callback
from dask.delayed import delayed
from dask.utils import tmpdir
from distributed import Client, Nanny, Variable
from distributed.utils_test import cluster, loop  # noqa
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

import dask_ml.model_selection as dcv
from dask_ml._compat import DISTRIBUTED_2_11_0, SK_0_23_2, WINDOWS
from dask_ml.model_selection import check_cv, compute_n_splits
from dask_ml.model_selection._search import _normalize_n_jobs
from dask_ml.model_selection.methods import CVCache
from dask_ml.model_selection.utils_test import (
    AsCompletedEstimator,
    CheckXClassifier,
    FailingClassifier,
    MockClassifier,
    MockClassifierWithFitParam,
    ScalingTransformer,
)


def _passthrough_scorer(estimator, *args, **kwargs):
    """Function that wraps estimator.score"""
    return estimator.score(*args, **kwargs)


class assert_dask_compute(Callback):
    def __init__(self, compute=False):
        self.compute = compute

    def __enter__(self):
        self.ran = False
        super(assert_dask_compute, self).__enter__()

    def __exit__(self, *args):
        if not self.compute and self.ran:
            raise ValueError("Unexpected call to compute")
        elif self.compute and not self.ran:
            raise ValueError("Expected call to compute, but none happened")
        super(assert_dask_compute, self).__exit__(*args)

    def _start(self, dsk):
        self.ran = True


def test_visualize():
    pytest.importorskip("graphviz")

    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = SVC(random_state=0, gamma="auto")
    grid = {"C": [0.1, 0.5, 0.9]}
    gs = dcv.GridSearchCV(clf, param_grid=grid).fit(X, y)

    assert hasattr(gs, "dask_graph_")

    with tmpdir() as d:
        gs.visualize(filename=os.path.join(d, "mydask"))
        assert os.path.exists(os.path.join(d, "mydask.png"))

    # Doesn't work if not fitted
    gs = dcv.GridSearchCV(clf, param_grid=grid)
    with pytest.raises(NotFittedError):
        gs.visualize()


np_X = np.random.normal(size=(20, 3))
np_y = np.random.randint(2, size=20)
np_groups = np.random.permutation(list(range(5)) * 4)
da_X = da.from_array(np_X, chunks=(3, 3))
da_y = da.from_array(np_y, chunks=3)
da_groups = da.from_array(np_groups, chunks=3)
del_X = delayed(np_X)
del_y = delayed(np_y)
del_groups = delayed(np_groups)


@pytest.mark.parametrize(
    ["cls", "has_shuffle"],
    [
        (KFold, True),
        (GroupKFold, False),
        (StratifiedKFold, True),
        (TimeSeriesSplit, False),
    ],
)
def test_kfolds(cls, has_shuffle):
    assert tokenize(cls(n_splits=3)) == tokenize(cls(n_splits=3))
    assert tokenize(cls(n_splits=3)) != tokenize(cls(n_splits=4))
    if has_shuffle:
        assert tokenize(cls(shuffle=True, random_state=0, n_splits=3)) == tokenize(
            cls(shuffle=True, random_state=0, n_splits=3)
        )

        rs = np.random.RandomState(42)
        assert tokenize(cls(shuffle=True, random_state=rs, n_splits=3)) == tokenize(
            cls(shuffle=True, random_state=rs, n_splits=3)
        )

        assert tokenize(cls(shuffle=True, random_state=0, n_splits=3)) != tokenize(
            cls(shuffle=True, random_state=2, n_splits=3)
        )

        assert tokenize(cls(shuffle=False, random_state=None, n_splits=3)) == tokenize(
            cls(shuffle=False, random_state=None, n_splits=3)
        )

    cv = cls(n_splits=3)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == 3

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == 3


@pytest.mark.parametrize(
    "cls", [ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit]
)
def test_shuffle_split(cls):
    assert tokenize(cls(n_splits=3, random_state=0)) == tokenize(
        cls(n_splits=3, random_state=0)
    )

    assert tokenize(cls(n_splits=3, random_state=0)) != tokenize(
        cls(n_splits=3, random_state=2)
    )

    assert tokenize(cls(n_splits=3, random_state=0)) != tokenize(
        cls(n_splits=4, random_state=0)
    )

    cv = cls(n_splits=3)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == 3

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == 3


@pytest.mark.parametrize("cvs", [(LeaveOneOut(),), (LeavePOut(2), LeavePOut(3))])
def test_leave_out(cvs):
    tokens = []
    for cv in cvs:
        assert tokenize(cv) == tokenize(cv)
        tokens.append(cv)
    assert len(set(tokens)) == len(tokens)

    cv = cvs[0]
    sol = cv.get_n_splits(np_X, np_y, np_groups)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == sol

    with assert_dask_compute(True):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == sol

    with assert_dask_compute(False):
        assert compute_n_splits(cv, np_X, da_y, da_groups) == sol


@pytest.mark.parametrize(
    "cvs", [(LeaveOneGroupOut(),), (LeavePGroupsOut(2), LeavePGroupsOut(3))]
)
def test_leave_group_out(cvs):
    tokens = []
    for cv in cvs:
        assert tokenize(cv) == tokenize(cv)
        tokens.append(cv)
    assert len(set(tokens)) == len(tokens)

    cv = cvs[0]
    sol = cv.get_n_splits(np_X, np_y, np_groups)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == sol

    with assert_dask_compute(True):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == sol

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, np_groups) == sol


def test_predefined_split():
    cv = PredefinedSplit(np.array(list(range(4)) * 5))
    cv2 = PredefinedSplit(np.array(list(range(5)) * 4))
    assert tokenize(cv) == tokenize(cv)
    assert tokenize(cv) != tokenize(cv2)

    sol = cv.get_n_splits(np_X, np_y, np_groups)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == sol

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == sol


def test_old_style_cv():
    cv1 = _CVIterableWrapper(
        [
            np.array([True, False, True, False] * 5),
            np.array([False, True, False, True] * 5),
        ]
    )
    cv2 = _CVIterableWrapper(
        [
            np.array([True, False, True, False] * 5),
            np.array([False, True, True, True] * 5),
        ]
    )
    assert tokenize(cv1) == tokenize(cv1)
    assert tokenize(cv1) != tokenize(cv2)

    sol = cv1.get_n_splits(np_X, np_y, np_groups)
    assert compute_n_splits(cv1, np_X, np_y, np_groups) == sol
    with assert_dask_compute(False):
        assert compute_n_splits(cv1, da_X, da_y, da_groups) == sol


def test_check_cv():
    # No y, classifier=False
    cv = check_cv(3, classifier=False)
    assert isinstance(cv, KFold) and cv.n_splits == 3
    cv = check_cv(5, classifier=False)
    assert isinstance(cv, KFold) and cv.n_splits == 5

    # y, classifier = False
    dy = da.from_array(np.array([1, 0, 1, 0, 1]), chunks=2)
    with assert_dask_compute(False):
        assert isinstance(check_cv(y=dy, classifier=False), KFold)

    # Binary and multi-class y
    for y in [
        np.array([0, 1, 0, 1, 0, 0, 1, 1, 1]),
        np.array([0, 1, 0, 1, 2, 1, 2, 0, 2]),
    ]:
        cv = check_cv(5, y, classifier=True)
        assert isinstance(cv, StratifiedKFold) and cv.n_splits == 5

        dy = da.from_array(y, chunks=2)
        with assert_dask_compute(True):
            cv = check_cv(5, dy, classifier=True)
        assert isinstance(cv, StratifiedKFold) and cv.n_splits == 5

    # Non-binary/multi-class y
    y = np.array([[1, 2], [0, 3], [0, 0], [3, 1], [2, 0]])
    assert isinstance(check_cv(y=y, classifier=True), KFold)

    dy = da.from_array(y, chunks=2)
    with assert_dask_compute(True):
        assert isinstance(check_cv(y=dy, classifier=True), KFold)

    # Old style
    cv = [np.array([True, False, True]), np.array([False, True, False])]
    with assert_dask_compute(False):
        assert isinstance(check_cv(cv, y=dy, classifier=True), _CVIterableWrapper)

    # CV instance passes through
    y = da.ones(5, chunks=2)
    cv = ShuffleSplit()
    with assert_dask_compute(False):
        assert check_cv(cv, y, classifier=True) is cv
        assert check_cv(cv, y, classifier=False) is cv


def test_grid_search_dask_inputs():
    # Numpy versions
    np_X, np_y = make_classification(n_samples=15, n_classes=2, random_state=0)
    np_groups = np.random.RandomState(0).randint(0, 3, 15)
    # Dask array versions
    da_X = da.from_array(np_X, chunks=5)
    da_y = da.from_array(np_y, chunks=5)
    da_groups = da.from_array(np_groups, chunks=5)
    # Delayed versions
    del_X = delayed(np_X)
    del_y = delayed(np_y)
    del_groups = delayed(np_groups)

    cv = GroupKFold(n_splits=3)
    clf = SVC(random_state=0, gamma="auto")
    grid = {"C": [1]}

    sol = SVC(C=1, random_state=0, gamma="auto").fit(np_X, np_y).support_vectors_

    for X, y, groups in product(
        [np_X, da_X, del_X], [np_y, da_y, del_y], [np_groups, da_groups, del_groups]
    ):
        gs = dcv.GridSearchCV(clf, param_grid=grid, cv=cv)

        with pytest.raises(ValueError) as exc:
            gs.fit(X, y)
        assert "parameter should not be None" in str(exc.value)

        gs.fit(X, y, groups=groups)
        np.testing.assert_allclose(sol, gs.best_estimator_.support_vectors_)


def test_grid_search_dask_dataframe():
    iris = load_iris()
    X = iris.data
    y = iris.target

    df = pd.DataFrame(X)
    ddf = dd.from_pandas(df, 2)

    dy = pd.Series(y)
    ddy = dd.from_pandas(dy, 2)

    clf = LogisticRegression(multi_class="auto", solver="lbfgs", max_iter=200)

    param_grid = {"C": [0.1, 1, 10]}
    gs = GridSearchCV(clf, param_grid, cv=5)
    dgs = dcv.GridSearchCV(clf, param_grid, cv=5)
    gs.fit(df, dy)
    dgs.fit(ddf, ddy)

    assert gs.best_params_ == dgs.best_params_


def test_pipeline_feature_union():
    iris = load_iris()
    X, y = iris.data, iris.target

    pca = PCA(random_state=0)
    kbest = SelectKBest()

    empty_union = FeatureUnion([("first", "drop"), ("second", "drop")])
    empty_pipeline = Pipeline([("first", None), ("second", None)])
    scaling = Pipeline([("transform", ScalingTransformer())])
    svc = SVC(kernel="linear", random_state=0)

    pipe = Pipeline(
        [
            ("empty_pipeline", empty_pipeline),
            ("scaling", scaling),
            ("missing", None),
            (
                "union",
                FeatureUnion(
                    [
                        ("pca", pca),
                        ("missing", "drop"),
                        ("kbest", kbest),
                        ("empty_union", empty_union),
                    ],
                    transformer_weights={"pca": 0.5},
                ),
            ),
            ("svc", svc),
        ]
    )

    param_grid = dict(
        scaling__transform__factor=[1, 2],
        union__pca__n_components=[1, 2, 3],
        union__kbest__k=[1, 2],
        svc__C=[0.1, 1, 10],
    )

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    gs.fit(X, y)
    dgs = dcv.GridSearchCV(pipe, param_grid=param_grid, scheduler="sync", cv=3)
    dgs.fit(X, y)

    # Check best params match
    assert gs.best_params_ == dgs.best_params_

    # Check PCA components match
    sk_pca = gs.best_estimator_.named_steps["union"].transformer_list[0][1]
    dk_pca = dgs.best_estimator_.named_steps["union"].transformer_list[0][1]
    np.testing.assert_allclose(sk_pca.components_, dk_pca.components_)

    # Check SelectKBest scores match
    sk_kbest = gs.best_estimator_.named_steps["union"].transformer_list[2][1]
    dk_kbest = dgs.best_estimator_.named_steps["union"].transformer_list[2][1]
    np.testing.assert_allclose(sk_kbest.scores_, dk_kbest.scores_)

    # Check SVC coefs match
    np.testing.assert_allclose(
        gs.best_estimator_.named_steps["svc"].coef_,
        dgs.best_estimator_.named_steps["svc"].coef_,
    )


def test_pipeline_sub_estimators():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaling = Pipeline([("transform", ScalingTransformer())])

    pipe = Pipeline(
        [
            ("setup", None),
            ("missing", None),
            ("scaling", scaling),
            ("svc", SVC(kernel="linear", random_state=0)),
        ]
    )

    param_grid = [
        {"svc__C": [0.1, 0.1]},  # Duplicates to test culling
        {
            "setup": [None],
            "svc__C": [0.1, 1, 10],
            "scaling": [ScalingTransformer(), None],
        },
        {
            "setup": [SelectKBest()],
            "setup__k": [1, 2],
            "svc": [
                SVC(kernel="linear", random_state=0, C=0.1),
                SVC(kernel="linear", random_state=0, C=1),
                SVC(kernel="linear", random_state=0, C=10),
            ],
        },
    ]

    gs = GridSearchCV(pipe, param_grid=param_grid, return_train_score=True, cv=3,)
    gs.fit(X, y)
    dgs = dcv.GridSearchCV(
        pipe, param_grid=param_grid, scheduler="sync", return_train_score=True, cv=3
    )
    dgs.fit(X, y)

    # Check best params match
    assert gs.best_params_ == dgs.best_params_

    # Check cv results match
    res = pd.DataFrame(dgs.cv_results_)
    sol = pd.DataFrame(gs.cv_results_)

    # TODO: Failures on Py36 / sklearn dev with order here.
    res = res.reindex(columns=sol.columns)

    pd.util.testing.assert_index_equal(res.columns, sol.columns)
    skip = ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"]
    res = res.drop(skip, axis=1)
    sol = sol.drop(skip, axis=1)
    pd.util.testing.assert_frame_equal(
        res, sol, check_exact=False, check_less_precise=1
    )

    # Check SVC coefs match
    np.testing.assert_allclose(
        gs.best_estimator_.named_steps["svc"].coef_,
        dgs.best_estimator_.named_steps["svc"].coef_,
    )


def check_scores_all_nan(gs, bad_param, score_key="score"):
    bad_param = "param_" + bad_param
    n_candidates = len(gs.cv_results_["params"])
    keys = ["split{}_test_{}".format(s, score_key) for s in range(gs.n_splits_)]
    assert all(
        np.isnan([gs.cv_results_[key][cand_i] for key in keys]).all()
        for cand_i in range(n_candidates)
        if gs.cv_results_[bad_param][cand_i] == FailingClassifier.FAILING_PARAMETER
    )


@pytest.mark.xfail(SK_0_23_2, reason="https://github.com/dask/dask-ml/issues/672")
@pytest.mark.parametrize(
    "weights", [None, (None, {"tr0": 2, "tr2": 3}, {"tr0": 2, "tr2": 4})]
)
def test_feature_union(weights):
    X = np.ones((10, 5))
    y = np.zeros(10)

    union = FeatureUnion(
        [
            ("tr0", ScalingTransformer()),
            ("tr1", ScalingTransformer()),
            ("tr2", ScalingTransformer()),
        ]
    )

    factors = [(2, 3, 5), (2, 4, 5), (2, 4, 6), (2, 4, None), (None, None, None)]
    params, sols, grid = [], [], []
    for constants, w in product(factors, weights or [None]):
        p = {}
        for n, c in enumerate(constants):
            if c is None:
                p["tr%d" % n] = None
            elif n == 3:  # 3rd is always an estimator
                p["tr%d" % n] = ScalingTransformer(c)
            else:
                p["tr%d__factor" % n] = c
        sol = union.set_params(transformer_weights=w, **p).transform(X)
        sols.append(sol)
        if w is not None:
            p["transformer_weights"] = w
        params.append(p)
        p2 = {"union__" + k: [v] for k, v in p.items()}
        p2["est"] = [CheckXClassifier(sol[0])]
        grid.append(p2)

    # Need to recreate the union after setting estimators to `None` above
    union = FeatureUnion(
        [
            ("tr0", ScalingTransformer()),
            ("tr1", ScalingTransformer()),
            ("tr2", ScalingTransformer()),
        ]
    )

    pipe = Pipeline([("union", union), ("est", CheckXClassifier())])
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False, cv=2, n_jobs=1)

    gs.fit(X, y)


def test_feature_union_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline(
        [
            (
                "union",
                FeatureUnion(
                    [("good", MockClassifier()), ("bad", FailingClassifier())],
                    transformer_weights={"bad": 0.5},
                ),
            ),
            ("clf", MockClassifier()),
        ]
    )

    grid = {"union__bad__parameter": [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False, scoring=None)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float("nan")
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)
    check_scores_all_nan(gs, "union__bad__parameter")


def test_feature_union_fit_failure_multiple_metrics():
    scoring = {"score_1": _passthrough_scorer, "score_2": _passthrough_scorer}
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline(
        [
            (
                "union",
                FeatureUnion(
                    [("good", MockClassifier()), ("bad", FailingClassifier())],
                    transformer_weights={"bad": 0.5},
                ),
            ),
            ("clf", MockClassifier()),
        ]
    )

    grid = {"union__bad__parameter": [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False, scoring=scoring)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float("nan")
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    for key in scoring:
        check_scores_all_nan(gs, "union__bad__parameter", score_key=key)


def test_failing_classifier_fails():
    clf = dcv.GridSearchCV(
        FailingClassifier(),
        {
            "parameter": [
                FailingClassifier.FAILING_PARAMETER,
                FailingClassifier.FAILING_SCORE_PARAMETER,
            ]
        },
        refit=False,
        return_train_score=False,
    )

    X, y = make_classification()

    with pytest.raises(ValueError, match="Failing"):
        clf.fit(X, y)

    clf = clf.set_params(error_score=-1)

    with pytest.warns(FitFailedWarning):
        clf.fit(X, y)

    for result in ["mean_fit_time", "mean_score_time", "mean_test_score"]:
        assert not any(np.isnan(clf.cv_results_[result]))


def test_pipeline_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline(
        [
            ("bad", FailingClassifier()),
            ("good1", MockClassifier()),
            ("good2", MockClassifier()),
        ]
    )

    grid = {
        "bad__parameter": [
            0,
            FailingClassifier.FAILING_PARAMETER,
            FailingClassifier.FAILING_PREDICT_PARAMETER,
            FailingClassifier.FAILING_SCORE_PARAMETER,
        ]
    }
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float("nan")
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    check_scores_all_nan(gs, "bad__parameter")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("in_pipeline", [False, True])
def test_estimator_predict_failure(in_pipeline):
    X, y = make_classification()
    if in_pipeline:
        clf = Pipeline([("bad", FailingClassifier())])
        key = "bad__parameter"
    else:
        clf = FailingClassifier()
        key = "parameter"

    grid = {
        key: [
            0,
            FailingClassifier.FAILING_PARAMETER,
            FailingClassifier.FAILING_PREDICT_PARAMETER,
            FailingClassifier.FAILING_SCORE_PARAMETER,
        ]
    }
    gs = dcv.GridSearchCV(
        clf, param_grid=grid, refit=False, error_score=float("nan"), cv=2
    )
    gs.fit(X, y)


def test_pipeline_raises():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([("step1", MockClassifier()), ("step2", MockClassifier())])

    grid = {"step3__parameter": [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False)
    with pytest.raises(ValueError):
        gs.fit(X, y)

    grid = {"steps": [[("one", MockClassifier()), ("two", MockClassifier())]]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False)
    with pytest.raises(NotImplementedError):
        gs.fit(X, y)


def test_feature_union_raises():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    union = FeatureUnion([("tr0", MockClassifier()), ("tr1", MockClassifier())])
    pipe = Pipeline([("union", union), ("est", MockClassifier())])

    grid = {"union__tr2__parameter": [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False)
    with pytest.raises(ValueError):
        gs.fit(X, y)

    grid = {"union__transformer_list": [[("one", MockClassifier())]]}
    gs = dcv.GridSearchCV(pipe, param_grid=grid, refit=False)
    with pytest.raises(NotImplementedError):
        gs.fit(X, y)


def test_bad_error_score():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    gs = dcv.GridSearchCV(
        MockClassifier(), {"foo_param": [0, 1, 2]}, error_score="badparam"
    )

    with pytest.raises(ValueError):
        gs.fit(X, y)


class CountTakes(np.ndarray):
    count = 0

    def take(self, *args, **kwargs):
        self.count += 1
        return super().take(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        self.count += 1
        return super().__getitem__(*args, **kwargs)


def test_cache_cv():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    X2 = X.view(CountTakes)
    gs = dcv.GridSearchCV(
        MockClassifier(),
        {"foo_param": [0, 1, 2]},
        cv=3,
        cache_cv=False,
        scheduler="sync",
    )
    gs.fit(X2, y)
    assert X2.count == 2 * 3 * 3  # (1 train + 1 test) * n_params * n_splits

    X2 = X.view(CountTakes)
    assert X2.count == 0
    gs.cache_cv = True
    gs.fit(X2, y)
    assert X2.count == 2 * 3  # (1 test + 1 train) * n_splits


def test_CVCache_serializable():
    inds = np.arange(10)
    splits = [(inds[:3], inds[3:]), (inds[3:], inds[:3])]
    X = np.arange(100).reshape((10, 10))
    y = np.zeros(10)
    cache = CVCache(splits, pairwise=True, cache=True)

    # Add something to the cache
    r1 = cache.extract(X, y, 0)
    assert cache.extract(X, y, 0) is r1
    assert len(cache.cache) == 1

    cache2 = pickle.loads(pickle.dumps(cache))
    assert len(cache2.cache) == 0
    assert cache2.pairwise == cache.pairwise
    assert all(
        (cache2.splits[i][j] == cache.splits[i][j]).all()
        for i in range(2)
        for j in range(2)
    )


def test_normalize_n_jobs():
    assert _normalize_n_jobs(-1) is None
    assert _normalize_n_jobs(-2) == cpu_count() - 1
    with pytest.raises(TypeError):
        _normalize_n_jobs("not an integer")


@pytest.mark.parametrize(
    "scheduler,n_jobs",
    [
        (None, 4),
        ("threading", 4),
        ("threading", 1),
        ("synchronous", 4),
        ("sync", 4),
        ("multiprocessing", 4),
        pytest.param(dask.get, 4, marks=[pytest.mark.filterwarnings("ignore")]),
    ],
)
def test_scheduler_param(scheduler, n_jobs):
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    gs = dcv.GridSearchCV(
        MockClassifier(),
        {"foo_param": [0, 1, 2]},
        cv=3,
        scheduler=scheduler,
        n_jobs=n_jobs,
    )
    gs.fit(X, y)


def test_scheduler_param_distributed(loop):  # noqa
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as client:
            gs = dcv.GridSearchCV(MockClassifier(), {"foo_param": [0, 1, 2]}, cv=3)
            gs.fit(X, y)

            def f(dask_scheduler):
                return len(dask_scheduler.transition_log)

            assert client.run_on_scheduler(f)  # some work happened on cluster


@pytest.mark.skipif(
    WINDOWS, reason="https://github.com/dask/dask-ml/issues/611 TimeoutError"
)
def test_as_completed_distributed(loop):  # noqa
    cluster_kwargs = dict(active_rpc_timeout=10, nanny=Nanny)
    if DISTRIBUTED_2_11_0:
        cluster_kwargs["disconnect_timeout"] = 10
    with cluster(**cluster_kwargs) as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            counter_name = "counter_name"
            counter = Variable(counter_name, client=c)
            counter.set(0)
            lock_name = "lock"

            killed_workers_name = "killed_workers"
            killed_workers = Variable(killed_workers_name, client=c)
            killed_workers.set({})

            X, y = make_classification(n_samples=100, n_features=10, random_state=0)
            gs = dcv.GridSearchCV(
                AsCompletedEstimator(killed_workers_name, lock_name, counter_name, 7),
                param_grid={"foo_param": [0, 1, 2]},
                cv=3,
                refit=False,
                cache_cv=False,
                scheduler=c,
            )
            gs.fit(X, y)

            def f(dask_scheduler):
                return dask_scheduler.transition_log

            def check_reprocess(transition_log):
                finished = set()
                for transition in transition_log:
                    key, start_state, end_state = (
                        transition[0],
                        transition[1],
                        transition[2],
                    )
                    assert key not in finished
                    if (
                        "score" in key
                        and start_state == "memory"
                        and end_state == "forgotten"
                    ):
                        finished.add(key)

            check_reprocess(c.run_on_scheduler(f))


def test_cv_multiplemetrics():
    X, y = make_classification(random_state=0)

    param_grid = {"max_depth": [1, 5]}
    a = dcv.GridSearchCV(
        RandomForestClassifier(n_estimators=10),
        param_grid,
        refit="score1",
        scoring={"score1": "accuracy", "score2": "accuracy"},
        cv=3,
    )
    b = GridSearchCV(
        RandomForestClassifier(n_estimators=10),
        param_grid,
        refit="score1",
        scoring={"score1": "accuracy", "score2": "accuracy"},
        cv=3,
    )
    a.fit(X, y)
    b.fit(X, y)

    assert a.best_score_ > 0
    assert isinstance(a.best_index_, type(b.best_index_))
    assert isinstance(a.best_params_, type(b.best_params_))


def test_cv_multiplemetrics_requires_refit_metric():
    X, y = make_classification(random_state=0)

    param_grid = {"max_depth": [1, 5]}
    a = dcv.GridSearchCV(
        RandomForestClassifier(n_estimators=10),
        param_grid,
        refit=True,
        scoring={"score1": "accuracy", "score2": "accuracy"},
    )

    with pytest.raises(ValueError):
        a.fit(X, y)


def test_cv_multiplemetrics_no_refit():
    X, y = make_classification(random_state=0)

    param_grid = {"max_depth": [1, 5]}
    a = dcv.GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        refit=False,
        scoring={"score1": "accuracy", "score2": "accuracy"},
    )
    b = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        refit=False,
        scoring={"score1": "accuracy", "score2": "accuracy"},
    )
    assert hasattr(a, "best_index_") is hasattr(b, "best_index_")
    assert hasattr(a, "best_estimator_") is hasattr(b, "best_estimator_")
    assert hasattr(a, "best_score_") is hasattr(b, "best_score_")


@pytest.mark.parametrize("cache_cv", [True, False])
def test_gridsearch_with_arraylike_fit_param(cache_cv):
    # https://github.com/dask/dask-ml/issues/319
    X, y = make_classification(random_state=0)
    param_grid = {"foo_param": [0.0001, 0.1]}

    a = dcv.GridSearchCV(
        MockClassifierWithFitParam(), param_grid, cv=3, refit=False, cache_cv=cache_cv,
    )
    b = GridSearchCV(MockClassifierWithFitParam(), param_grid, cv=3, refit=False)

    b.fit(X, y, mock_fit_param=[0, 1])
    a.fit(X, y, mock_fit_param=[0, 1])


def test_mock_with_fit_param_raises():
    X, y = make_classification(random_state=0)

    clf = MockClassifierWithFitParam()

    with pytest.raises(ValueError):
        clf.fit(X, y)
