# NOTE: These tests were copied (with modification) from the equivalent
# scikit-learn testing code. The scikit-learn license has been included at
# dask_ml/licences/COPY

import pickle

import dask
import dask.array as da
import numpy as np
import packaging.version
import pytest
import scipy.sparse as sp
import sklearn.metrics
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from scipy.stats import expon
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
)
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dask_ml import model_selection as dcv
from dask_ml._compat import SK_VERSION
from dask_ml.model_selection.utils_test import (
    CheckingClassifier,
    FailingClassifier,
    MockClassifier,
    MockDataFrame,
)


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""

    @property
    def score(self):
        raise AttributeError


rng = np.random.RandomState(0)


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

da_X = da.from_array(rng.normal(size=(20, 3)), chunks=(3, 3))
da_y = da.from_array(rng.randint(2, size=20), chunks=3)


def assert_grid_iter_equals_getitem(grid):
    assert list(grid) == [grid[i] for i in range(len(grid))]


def test_grid_search():
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]})
    # make sure it selects the smallest parameter in case of ties
    grid_search.fit(X, y)
    assert grid_search.best_estimator_.foo_param == 2

    assert_array_equal(grid_search.cv_results_["param_foo_param"].data, [1, 2, 3])

    # Smoke test the score etc:
    grid_search.score(X, y)
    grid_search.predict_proba(X)
    grid_search.decision_function(X)
    grid_search.transform(X)

    # Test exception handling on scoring
    grid_search.scoring = "sklearn"
    with pytest.raises(ValueError):
        grid_search.fit(X, y)


@pytest.mark.parametrize(
    "cls,kwargs", [(dcv.GridSearchCV, {}), (dcv.RandomizedSearchCV, {"n_iter": 1})]
)
def test_hyperparameter_searcher_with_fit_params(cls, kwargs):
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_fit_params=["spam", "eggs"])
    pipe = Pipeline([("clf", clf)])
    searcher = cls(pipe, {"clf__foo_param": [1, 2, 3]}, cv=2, **kwargs)

    # The CheckingClassifer generates an assertion error if
    # a parameter is missing or has length != len(X).
    with pytest.raises(AssertionError) as exc:
        searcher.fit(X, y, clf__spam=np.ones(10))
    assert "Expected fit parameter(s) ['eggs'] not seen." in str(exc.value)

    searcher.fit(X, y, clf__spam=np.ones(10), clf__eggs=np.zeros(10))
    # Test with dask objects as parameters
    searcher.fit(
        X, y, clf__spam=da.ones(10, chunks=2), clf__eggs=dask.delayed(np.zeros(10))
    )


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_grid_search_no_score():
    # Test grid-search on classifier that has no score function.
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    clf_no_score = LinearSVCNoScore(random_state=0)

    # XXX: It seems there's some global shared state in LinearSVC - fitting
    # multiple `SVC` instances in parallel using threads sometimes results in
    # wrong results. This only happens with threads, not processes/sync.
    # For now, we'll fit using the sync scheduler.
    grid_search = dcv.GridSearchCV(clf, {"C": Cs}, scoring="accuracy", scheduler="sync")
    grid_search.fit(X, y)

    grid_search_no_score = dcv.GridSearchCV(
        clf_no_score, {"C": Cs}, scoring="accuracy", scheduler="sync"
    )
    # smoketest grid search
    grid_search_no_score.fit(X, y)

    # check that best params are equal
    assert grid_search_no_score.best_params_ == grid_search.best_params_
    # check that we can call score and that it gives the correct result
    assert grid_search.score(X, y) == grid_search_no_score.score(X, y)

    # giving no scoring function raises an error
    grid_search_no_score = dcv.GridSearchCV(clf_no_score, {"C": Cs})
    with pytest.raises(TypeError) as exc:
        grid_search_no_score.fit([[1]])
    assert "no scoring" in str(exc.value)


def test_grid_search_score_method():
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {"C": [0.1]}

    search_no_scoring = dcv.GridSearchCV(clf, grid, scoring=None).fit(X, y)
    search_accuracy = dcv.GridSearchCV(clf, grid, scoring="accuracy").fit(X, y)
    search_no_score_method_auc = dcv.GridSearchCV(
        LinearSVCNoScore(), grid, scoring="roc_auc"
    ).fit(X, y)
    search_auc = dcv.GridSearchCV(clf, grid, scoring="roc_auc").fit(X, y)

    # Check warning only occurs in situation where behavior changed:
    # estimator requires score method to compete with scoring parameter
    score_no_scoring = search_no_scoring.score(X, y)
    score_accuracy = search_accuracy.score(X, y)
    score_no_score_auc = search_no_score_method_auc.score(X, y)
    score_auc = search_auc.score(X, y)

    # ensure the test is sane
    assert score_auc < 1.0
    assert score_accuracy < 1.0
    assert score_auc != score_accuracy

    assert_almost_equal(score_accuracy, score_no_scoring)
    assert_almost_equal(score_auc, score_no_score_auc)


def test_grid_search_groups():
    # Check if ValueError (when groups is None) propagates to dcv.GridSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 15)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(n_splits=3),
    ]
    for cv in group_cvs:
        gs = dcv.GridSearchCV(clf, grid, cv=cv)

        with pytest.raises(ValueError) as exc:
            assert gs.fit(X, y)
        assert "parameter should not be None" in str(exc.value)

        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(n_splits=3), StratifiedShuffleSplit(n_splits=3)]
    for cv in non_group_cvs:
        gs = dcv.GridSearchCV(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


@pytest.mark.xfail(reason="flaky test", strict=False)
def test_return_train_score_warn():
    # Test that warnings are raised. Will be removed in sklearn 0.21
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    X = (X - X.mean(0)) / X.std(0)  # help convergence
    grid = {"C": [0.1, 0.5]}

    for val in [True, False]:
        est = dcv.GridSearchCV(
            LinearSVC(random_state=0, tol=0.5), grid, return_train_score=val
        )
        with pytest.warns(None) as warns:
            results = est.fit(X, y).cv_results_
        assert not warns
        assert type(results) is dict

    est = dcv.GridSearchCV(LinearSVC(random_state=0), grid)
    with pytest.warns(None) as warns:
        results = est.fit(X, y).cv_results_
    assert not warns

    train_keys = {
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "mean_train_score",
        "std_train_score",
    }

    include_train_score = SK_VERSION <= packaging.version.parse("0.21.dev0")

    if include_train_score:
        assert all(x in results for x in train_keys)
    else:
        result = train_keys & set(results)
        assert result == {}

    for key in results:
        if key in train_keys:
            with pytest.warns(FutureWarning):
                results[key]
        else:
            with pytest.warns(None) as warns:
                results[key]
            assert not warns


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_classes__property():
    # Test that classes_ property matches best_estimator_.classes_
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    Cs = [0.1, 1, 10]

    grid_search = dcv.GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    grid_search.fit(X, y)
    assert_array_equal(grid_search.best_estimator_.classes_, grid_search.classes_)

    # Test that regressors do not have a classes_ attribute
    grid_search = dcv.GridSearchCV(Ridge(), {"alpha": [1.0, 2.0]})
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute before it's fit
    grid_search = dcv.GridSearchCV(LinearSVC(random_state=0), {"C": Cs})
    assert not hasattr(grid_search, "classes_")

    # Test that the grid searcher has no classes_ attribute without a refit
    grid_search = dcv.GridSearchCV(LinearSVC(random_state=0), {"C": Cs}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "classes_")


def test_trivial_cv_results_attr():
    # Test search over a "grid" with only one point.
    # Non-regression test: grid_scores_ wouldn't be set by dcv.GridSearchCV.
    clf = MockClassifier()
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1]})
    grid_search.fit(X, y)
    assert hasattr(grid_search, "cv_results_")

    random_search = dcv.RandomizedSearchCV(clf, {"foo_param": [0]}, n_iter=1)
    random_search.fit(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_no_refit():
    # Test that GSCV can be used for model selection alone without refitting
    clf = MockClassifier()
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=False)
    grid_search.fit(X, y)
    assert not hasattr(grid_search, "best_estimator_")
    assert not hasattr(grid_search, "best_index_")
    assert not hasattr(grid_search, "best_score_")
    assert not hasattr(grid_search, "best_params_")

    # Make sure the predict/transform etc fns raise meaningfull error msg
    for fn_name in (
        "predict",
        "predict_proba",
        "predict_log_proba",
        "transform",
        "inverse_transform",
    ):
        with pytest.raises(NotFittedError) as exc:
            getattr(grid_search, fn_name)(X)
        assert (
            "refit=False. %s is available only after refitting on the "
            "best parameters" % fn_name
        ) in str(exc.value)


def test_no_refit_multiple_metrics():
    clf = DecisionTreeClassifier()
    scoring = {"score_1": "accuracy", "score_2": "accuracy"}

    gs = dcv.GridSearchCV(clf, {"max_depth": [1, 2, 3]}, refit=False, scoring=scoring)
    gs.fit(da_X, da_y)
    assert not hasattr(gs, "best_estimator_")
    assert not hasattr(gs, "best_index_")
    assert not hasattr(gs, "best_score_")
    assert not hasattr(gs, "best_params_")

    for fn_name in ("predict", "predict_proba", "predict_log_proba"):
        with pytest.raises(NotFittedError) as exc:
            getattr(gs, fn_name)(X)
        assert (
            "refit=False. %s is available only after refitting on the "
            "best parameters" % fn_name
        ) in str(exc.value)


def test_grid_search_error():
    # Test that grid search will capture errors on data with different length
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(X_[:180], y_)


def test_grid_search_one_grid_point():
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)
    param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

    clf = SVC()
    cv = dcv.GridSearchCV(clf, param_dict)
    cv.fit(X_, y_)

    clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
    clf.fit(X_, y_)

    assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)


def test_grid_search_bad_param_grid():
    param_dict = {"C": 1.0}
    clf = SVC()

    with pytest.raises(ValueError):
        dcv.GridSearchCV(clf, param_dict)

    param_dict = {"C": []}
    clf = SVC()

    with pytest.raises(ValueError):
        dcv.GridSearchCV(clf, param_dict)

    param_dict = {"C": "1,2,3"}
    clf = SVC()

    with pytest.raises(ValueError):
        dcv.GridSearchCV(clf, param_dict)

    param_dict = {"C": np.ones(6).reshape(3, 2)}
    clf = SVC()
    with pytest.raises(ValueError):
        dcv.GridSearchCV(clf, param_dict)


def test_grid_search_sparse():
    # Test that grid search works with both dense and sparse matrices
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = sp.csr_matrix(X_)
    clf = LinearSVC()
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(X_[:180].tocoo(), y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert np.mean(y_pred == y_pred2) >= 0.9
    assert C == C2


def test_grid_search_sparse_scoring():
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator_.C

    X_ = sp.csr_matrix(X_)
    clf = LinearSVC()
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
    cv.fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator_.C

    assert_array_equal(y_pred, y_pred2)
    assert C == C2
    # Smoke test the score
    # np.testing.assert_allclose(f1_score(cv.predict(X_[:180]), y[:180]),
    #                            cv.score(X_[:180], y[:180]))

    # test loss where greater is worse
    def f1_loss(y_true_, y_pred_):
        return -f1_score(y_true_, y_pred_)

    F1Loss = make_scorer(f1_loss, greater_is_better=False)
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]}, scoring=F1Loss)
    cv.fit(X_[:180], y_[:180])
    y_pred3 = cv.predict(X_[180:])
    C3 = cv.best_estimator_.C

    assert C == C3
    assert_array_equal(y_pred, y_pred3)


def test_grid_search_precomputed_kernel():
    # Test that grid search works when the input features are given in the
    # form of a precomputed kernel matrix
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    # compute the training kernel matrix corresponding to the linear kernel
    K_train = np.dot(X_[:180], X_[:180].T)
    y_train = y_[:180]

    clf = SVC(kernel="precomputed")
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]})
    cv.fit(K_train, y_train)

    assert cv.best_score_ >= 0

    # compute the test kernel matrix
    K_test = np.dot(X_[180:], X_[:180].T)
    y_test = y_[180:]

    y_pred = cv.predict(K_test)

    assert np.mean(y_pred == y_test) >= 0

    # test error is raised when the precomputed kernel is not array-like
    # or sparse
    with pytest.raises(ValueError):
        cv.fit(K_train.tolist(), y_train)


def test_grid_search_precomputed_kernel_error_nonsquare():
    # Test that grid search returns an error with a non-square precomputed
    # training kernel matrix
    K_train = np.zeros((10, 20))
    y_train = np.ones((10,))
    clf = SVC(kernel="precomputed")
    cv = dcv.GridSearchCV(clf, {"C": [0.1, 1.0]})
    with pytest.raises(ValueError):
        cv.fit(K_train, y_train)


class BrokenClassifier(BaseEstimator):
    """Broken classifier that cannot be fit twice"""

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y):
        assert not hasattr(self, "has_been_fit_")
        self.has_been_fit_ = True

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_refit():
    # Regression test for bug in refitting
    # Simulates re-fitting a broken estimator; this used to break with
    # sparse SVMs.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = dcv.GridSearchCV(
        BrokenClassifier(), [{"parameter": [0, 1]}], scoring="accuracy", refit=True
    )
    clf.fit(X, y)


def test_gridsearch_nd():
    # Pass X as list in dcv.GridSearchCV
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
    clf = CheckingClassifier(
        check_X=lambda x: x.shape[1:] == (5, 3, 2),
        check_y=lambda x: x.shape[1:] == (7, 11),
    )
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]})
    grid_search.fit(X_4d, y_3d).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_X_as_list():
    # Pass X as list in dcv.GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(check_X=lambda x: isinstance(x, list))
    cv = KFold(n_splits=3)
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X.tolist(), y).score(X, y)
    assert hasattr(grid_search, "cv_results_")


def test_y_as_list():
    # Pass y as list in dcv.GridSearchCV
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = CheckingClassifier(check_y=lambda x: isinstance(x, list))
    cv = KFold(n_splits=3)
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
    grid_search.fit(X, y.tolist()).score(X, y)
    assert hasattr(grid_search, "cv_results_")


@pytest.mark.filterwarnings("ignore")
def test_pandas_input():
    # check cross_val_score doesn't destroy pandas dataframe
    types = [(MockDataFrame, MockDataFrame)]
    try:
        from pandas import Series, DataFrame

        types.append((DataFrame, Series))
    except ImportError:
        pass

    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    for InputFeatureType, TargetType in types:
        # X dataframe, y series
        X_df, y_ser = InputFeatureType(X), TargetType(y)
        clf = CheckingClassifier(
            check_X=lambda x: isinstance(x, InputFeatureType),
            check_y=lambda x: isinstance(x, TargetType),
        )

        grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]})
        grid_search.fit(X_df, y_ser).score(X_df, y_ser)
        grid_search.predict(X_df)
        assert hasattr(grid_search, "cv_results_")


def test_unsupervised_grid_search():
    # test grid-search with unsupervised estimator
    X, y = make_blobs(random_state=0)
    km = KMeans(random_state=0)
    grid_search = dcv.GridSearchCV(
        km, param_grid=dict(n_clusters=[2, 3, 4]), scoring="adjusted_rand_score"
    )
    grid_search.fit(X, y)
    # ARI can find the right number :)
    assert grid_search.best_params_["n_clusters"] == 3

    # Now without a score, and without y
    grid_search = dcv.GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]))
    grid_search.fit(X)
    assert grid_search.best_params_["n_clusters"] == 4


def test_gridsearch_no_predict():
    # test grid-search with an estimator without predict.
    # slight duplication of a test from KDE
    def custom_scoring(estimator, X):
        return 42 if estimator.bandwidth == 0.1 else 0

    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    search = dcv.GridSearchCV(
        KernelDensity(),
        param_grid=dict(bandwidth=[0.01, 0.1, 1]),
        scoring=custom_scoring,
    )
    search.fit(X)
    assert search.best_params_["bandwidth"] == 0.1
    assert search.best_score_ == 42


def check_cv_results_array_types(cv_results, param_keys, score_keys):
    # Check if the search `cv_results`'s array are of correct types
    assert all(isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys)
    assert all(cv_results[key].dtype == object for key in param_keys)
    assert not any(isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys)
    assert all(
        cv_results[key].dtype == np.float64
        for key in score_keys
        if not key.startswith("rank")
    )
    assert cv_results["rank_test_score"].dtype == np.int32


def check_cv_results_keys(cv_results, param_keys, score_keys, n_cand):
    # Test the search.cv_results_ contains all the required results
    assert_array_equal(
        sorted(cv_results.keys()), sorted(param_keys + score_keys + ("params",))
    )
    assert all(cv_results[key].shape == (n_cand,) for key in param_keys + score_keys)


def test_grid_search_cv_results():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)

    n_splits = 3
    n_grid_points = 6
    params = [
        dict(kernel=["rbf"], C=[1, 10], gamma=[0.1, 1]),
        dict(kernel=["poly"], degree=[1, 2]),
    ]
    grid_search = dcv.GridSearchCV(
        SVC(gamma="auto"),
        cv=n_splits,
        iid=False,
        param_grid=params,
        return_train_score=True,
    )
    grid_search.fit(X, y)
    grid_search_iid = dcv.GridSearchCV(
        SVC(gamma="auto"),
        cv=n_splits,
        iid=True,
        param_grid=params,
        return_train_score=True,
    )
    grid_search_iid.fit(X, y)

    param_keys = ("param_C", "param_degree", "param_gamma", "param_kernel")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_candidates = n_grid_points

    for search, iid in zip((grid_search, grid_search_iid), (False, True)):
        assert iid == search.iid
        cv_results = search.cv_results_
        # Check if score and timing are reasonable
        assert all(cv_results["rank_test_score"] >= 1)
        assert all(
            all(cv_results[k] >= 0) for k in score_keys if k != "rank_test_score"
        )
        assert all(
            all(cv_results[k] <= 1)
            for k in score_keys
            if "time" not in k and k != "rank_test_score"
        )
        # Check cv_results structure
        check_cv_results_array_types(cv_results, param_keys, score_keys)
        check_cv_results_keys(cv_results, param_keys, score_keys, n_candidates)
        # Check masking
        cv_results = grid_search.cv_results_
        n_candidates = len(grid_search.cv_results_["params"])
        assert all(
            (
                cv_results["param_C"].mask[i]
                and cv_results["param_gamma"].mask[i]
                and not cv_results["param_degree"].mask[i]
            )
            for i in range(n_candidates)
            if cv_results["param_kernel"][i] == "linear"
        )
        assert all(
            (
                not cv_results["param_C"].mask[i]
                and not cv_results["param_gamma"].mask[i]
                and cv_results["param_degree"].mask[i]
            )
            for i in range(n_candidates)
            if cv_results["param_kernel"][i] == "rbf"
        )


@pytest.mark.parametrize(
    "params",
    [
        {"C": expon(scale=10), "gamma": expon(scale=0.1)},
        [
            {"C": expon(scale=10), "gamma": expon(scale=0.1)},
            {"C": expon(scale=20), "gamma": expon(scale=0.2)},
        ],
    ],
)
def test_random_search_cv_results(params):
    # Make a dataset with a lot of noise to get various kind of prediction
    # errors across CV folds and parameter settings
    X, y = make_classification(
        n_samples=200, n_features=100, n_informative=3, random_state=0
    )

    # scipy.stats dists now supports `seed` but we still support scipy 0.12
    # which doesn't support the seed. Hence the assertions in the test for
    # random_search alone should not depend on randomization.
    n_splits = 3
    n_search_iter = 30
    random_search = dcv.RandomizedSearchCV(
        SVC(),
        n_iter=n_search_iter,
        cv=n_splits,
        iid=False,
        param_distributions=params,
        return_train_score=True,
    )
    random_search.fit(X, y)
    random_search_iid = dcv.RandomizedSearchCV(
        SVC(),
        n_iter=n_search_iter,
        cv=n_splits,
        iid=True,
        param_distributions=params,
        return_train_score=True,
    )
    random_search_iid.fit(X, y)

    param_keys = ("param_C", "param_gamma")
    score_keys = (
        "mean_test_score",
        "mean_train_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
        "split0_train_score",
        "split1_train_score",
        "split2_train_score",
        "std_test_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    )
    n_cand = n_search_iter

    for search, iid in zip((random_search, random_search_iid), (False, True)):
        assert iid == search.iid
        cv_results = search.cv_results_
        # Check results structure
        check_cv_results_array_types(cv_results, param_keys, score_keys)
        check_cv_results_keys(cv_results, param_keys, score_keys, n_cand)
        # For random_search, all the param array vals should be unmasked
        assert not (
            any(cv_results["param_C"].mask) or any(cv_results["param_gamma"].mask)
        )


def test_search_iid_param():
    # Test the IID parameter
    # noise-free simple 2d-data
    X, y = make_blobs(
        centers=[[0, 0], [1, 0], [0, 1], [1, 1]],
        random_state=0,
        cluster_std=0.1,
        shuffle=False,
        n_samples=80,
    )
    # split dataset into two folds that are not iid
    # first one contains data of all 4 blobs, second only from two.
    mask = np.ones(X.shape[0], dtype=np.bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0
    # this leads to perfect classification on one fold and a score of 1/3 on
    # the other
    # create "cv" for splits
    cv = [[mask, ~mask], [~mask, mask]]
    # once with iid=True (default)
    grid_search = dcv.GridSearchCV(
        SVC(gamma="auto"), param_grid={"C": [1, 10]}, cv=cv, return_train_score=True
    )
    random_search = dcv.RandomizedSearchCV(
        SVC(gamma="auto"),
        n_iter=2,
        param_distributions={"C": [1, 10]},
        return_train_score=True,
        cv=cv,
    )
    for search in (grid_search, random_search):
        search.fit(X, y)
        assert search.iid

        test_cv_scores = np.array(
            list(
                search.cv_results_["split%d_test_score" % s_i][0]
                for s_i in range(search.n_splits_)
            )
        )
        train_cv_scores = np.array(
            list(
                search.cv_results_["split%d_train_" "score" % s_i][0]
                for s_i in range(search.n_splits_)
            )
        )
        test_mean = search.cv_results_["mean_test_score"][0]
        test_std = search.cv_results_["std_test_score"][0]

        train_cv_scores = np.array(
            list(
                search.cv_results_["split%d_train_" "score" % s_i][0]
                for s_i in range(search.n_splits_)
            )
        )
        train_mean = search.cv_results_["mean_train_score"][0]
        train_std = search.cv_results_["std_train_score"][0]

        # Test the first candidate
        assert search.cv_results_["param_C"][0] == 1
        assert_array_almost_equal(test_cv_scores, [1, 1.0 / 3.0])
        assert_array_almost_equal(train_cv_scores, [1, 1])

        # for first split, 1/4 of dataset is in test, for second 3/4.
        # take weighted average and weighted std
        expected_test_mean = 1 * 1.0 / 4.0 + 1.0 / 3.0 * 3.0 / 4.0
        expected_test_std = np.sqrt(
            1.0 / 4 * (expected_test_mean - 1) ** 2
            + 3.0 / 4 * (expected_test_mean - 1.0 / 3.0) ** 2
        )
        assert_almost_equal(test_mean, expected_test_mean)
        assert_almost_equal(test_std, expected_test_std)

        # For the train scores, we do not take a weighted mean irrespective of
        # i.i.d. or not
        assert_almost_equal(train_mean, 1)
        assert_almost_equal(train_std, 0)

    # once with iid=False
    grid_search = dcv.GridSearchCV(
        SVC(gamma="auto"),
        param_grid={"C": [1, 10]},
        cv=cv,
        iid=False,
        return_train_score=True,
    )
    random_search = dcv.RandomizedSearchCV(
        SVC(gamma="auto"),
        n_iter=2,
        param_distributions={"C": [1, 10]},
        cv=cv,
        iid=False,
        return_train_score=True,
    )

    for search in (grid_search, random_search):
        search.fit(X, y)
        assert not search.iid

        test_cv_scores = np.array(
            list(
                search.cv_results_["split%d_test_score" % s][0]
                for s in range(search.n_splits_)
            )
        )
        test_mean = search.cv_results_["mean_test_score"][0]
        test_std = search.cv_results_["std_test_score"][0]

        train_cv_scores = np.array(
            list(
                search.cv_results_["split%d_train_" "score" % s][0]
                for s in range(search.n_splits_)
            )
        )
        train_mean = search.cv_results_["mean_train_score"][0]
        train_std = search.cv_results_["std_train_score"][0]

        assert search.cv_results_["param_C"][0] == 1
        # scores are the same as above
        assert_array_almost_equal(test_cv_scores, [1, 1.0 / 3.0])
        # Unweighted mean/std is used
        assert_almost_equal(test_mean, np.mean(test_cv_scores))
        assert_almost_equal(test_std, np.std(test_cv_scores))

        # For the train scores, we do not take a weighted mean irrespective of
        # i.i.d. or not
        assert_almost_equal(train_mean, 1)
        assert_almost_equal(train_std, 0)


def test_search_cv_results_rank_tie_breaking():
    X, y = make_blobs(n_samples=50, random_state=42)

    # The two C values are close enough to give similar models
    # which would result in a tie of their mean cv-scores
    param_grid = {"C": [1, 1.001, 0.001]}

    grid_search = dcv.GridSearchCV(
        SVC(gamma="auto"), param_grid=param_grid, return_train_score=True
    )
    random_search = dcv.RandomizedSearchCV(
        SVC(gamma="auto"),
        n_iter=3,
        param_distributions=param_grid,
        return_train_score=True,
    )

    for search in (grid_search, random_search):
        search.fit(X, y)
        cv_results = search.cv_results_
        # Check tie breaking strategy -
        # Check that there is a tie in the mean scores between
        # candidates 1 and 2 alone
        assert_almost_equal(
            cv_results["mean_test_score"][0], cv_results["mean_test_score"][1]
        )
        assert_almost_equal(
            cv_results["mean_train_score"][0], cv_results["mean_train_score"][1]
        )
        try:
            assert_almost_equal(
                cv_results["mean_test_score"][1], cv_results["mean_test_score"][2]
            )
        except AssertionError:
            pass
        try:
            assert_almost_equal(
                cv_results["mean_train_score"][1], cv_results["mean_train_score"][2]
            )
        except AssertionError:
            pass
        # 'min' rank should be assigned to the tied candidates
        assert_almost_equal(search.cv_results_["rank_test_score"], [1, 1, 3])


def test_search_cv_results_none_param():
    X, y = [[1], [2], [3], [4], [5]], [0, 0, 0, 0, 1]
    estimators = (DecisionTreeRegressor(), DecisionTreeClassifier())
    est_parameters = {"random_state": [0, None]}
    cv = KFold(random_state=0, n_splits=2, shuffle=True)

    for est in estimators:
        grid_search = dcv.GridSearchCV(est, est_parameters, cv=cv).fit(X, y)
        assert_array_equal(grid_search.cv_results_["param_random_state"], [0, None])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_grid_search_correct_score_results():
    # test that correct scores are used
    n_splits = 3
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    for score in ["f1", "roc_auc"]:
        # XXX: It seems there's some global shared state in LinearSVC - fitting
        # multiple `SVC` instances in parallel using threads sometimes results
        # in wrong results. This only happens with threads, not processes/sync.
        # For now, we'll fit using the sync scheduler.
        grid_search = dcv.GridSearchCV(
            clf, {"C": Cs}, scoring=score, cv=n_splits, scheduler="sync"
        )
        cv_results = grid_search.fit(X, y).cv_results_

        # Test scorer names
        result_keys = list(cv_results.keys())
        expected_keys = ("mean_test_score", "rank_test_score") + tuple(
            "split%d_test_score" % cv_i for cv_i in range(n_splits)
        )
        assert all(np.in1d(expected_keys, result_keys))

        cv = StratifiedKFold(n_splits=n_splits)
        n_splits = grid_search.n_splits_
        for candidate_i, C in enumerate(Cs):
            clf.set_params(C=C)
            cv_scores = np.array(
                list(
                    grid_search.cv_results_["split%d_test_score" % s][candidate_i]
                    for s in range(n_splits)
                )
            )
            for i, (train, test) in enumerate(cv.split(X, y)):
                clf.fit(X[train], y[train])
                if score == "f1":
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == "roc_auc":
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                assert_almost_equal(correct_score, cv_scores[i])


def test_pickle():
    # Test that a fit search can be pickled
    clf = MockClassifier()
    grid_search = dcv.GridSearchCV(clf, {"foo_param": [1, 2, 3]}, refit=True)
    grid_search.fit(X, y)
    grid_search_pickled = pickle.loads(pickle.dumps(grid_search))
    assert_array_almost_equal(grid_search.predict(X), grid_search_pickled.predict(X))

    random_search = dcv.RandomizedSearchCV(
        clf, {"foo_param": [1, 2, 3]}, refit=True, n_iter=3
    )
    random_search.fit(X, y)
    random_search_pickled = pickle.loads(pickle.dumps(random_search))
    assert_array_almost_equal(
        random_search.predict(X), random_search_pickled.predict(X)
    )


def test_grid_search_with_multioutput_data():
    # Test search with multi-output estimator

    X, y = make_multilabel_classification(return_indicator=True, random_state=0)

    est_parameters = {"max_depth": [1, 2, 3, 4]}
    cv = KFold(random_state=0, n_splits=3, shuffle=True)

    estimators = [
        DecisionTreeRegressor(random_state=0),
        DecisionTreeClassifier(random_state=0),
    ]

    scoring = sklearn.metrics.make_scorer(
        sklearn.metrics.roc_auc_score, average="weighted"
    )
    # Test with grid search cv
    for est in estimators:
        grid_search = dcv.GridSearchCV(est, est_parameters, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        res_params = grid_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = scoring(est, X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    grid_search.cv_results_["split%d_test_score" % i][cand_i],
                )

    # Test with a randomized search
    for est in estimators:
        random_search = dcv.RandomizedSearchCV(
            est, est_parameters, cv=cv, n_iter=3, scoring=scoring
        )
        random_search.fit(X, y)
        res_params = random_search.cv_results_["params"]
        for cand_i in range(len(res_params)):
            est.set_params(**res_params[cand_i])

            for i, (train, test) in enumerate(cv.split(X, y)):
                est.fit(X[train], y[train])
                correct_score = scoring(est, X[test], y[test])
                assert_almost_equal(
                    correct_score,
                    random_search.cv_results_["split%d_test_score" % i][cand_i],
                )


def test_predict_proba_disabled():
    # Test predict_proba when disabled on estimator.
    X = np.arange(20).reshape(5, -1)
    y = [0, 0, 1, 1, 1]
    clf = SVC(probability=False, gamma="auto")
    gs = dcv.GridSearchCV(clf, {}, cv=2).fit(X, y)
    assert not hasattr(gs, "predict_proba")


def test_grid_search_allows_nans():
    # Test dcv.GridSearchCV with Imputer
    X = np.arange(20, dtype=np.float64).reshape(5, -1)
    X[2, :] = np.nan
    y = [0, 0, 1, 1, 1]

    imputer = SimpleImputer(strategy="mean", missing_values=np.nan)
    p = Pipeline([("imputer", imputer), ("classifier", MockClassifier())])
    dcv.GridSearchCV(p, {"classifier__foo_param": [1, 2, 3]}, cv=2).fit(X, y)


@pytest.mark.filterwarnings("ignore")
def test_grid_search_failing_classifier():
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)
    clf = FailingClassifier()

    # refit=False because we want to test the behaviour of the grid search part
    gs = dcv.GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=0.0,
    )

    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    n_candidates = len(gs.cv_results_["params"])

    # Ensure that grid scores were set to zero as required for those fits
    # that are expected to fail.
    def get_cand_scores(i):
        return np.array(
            list(
                gs.cv_results_["split%d_test_score" % s][i] for s in range(gs.n_splits_)
            )
        )

    assert all(
        (
            np.all(get_cand_scores(cand_i) == 0.0)
            for cand_i in range(n_candidates)
            if gs.cv_results_["param_parameter"][cand_i]
            == FailingClassifier.FAILING_PARAMETER
        )
    )

    gs = dcv.GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score=float("nan"),
    )

    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    n_candidates = len(gs.cv_results_["params"])
    assert all(
        np.all(np.isnan(get_cand_scores(cand_i)))
        for cand_i in range(n_candidates)
        if gs.cv_results_["param_parameter"][cand_i]
        == FailingClassifier.FAILING_PARAMETER
    )


def test_grid_search_failing_classifier_raise():
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)
    clf = FailingClassifier()

    # refit=False because we want to test the behaviour of the grid search part
    gs = dcv.GridSearchCV(
        clf,
        [{"parameter": [0, 1, 2]}],
        scoring="accuracy",
        refit=False,
        error_score="raise",
    )

    # FailingClassifier issues a ValueError so this is what we look for.
    with pytest.raises(ValueError):
        gs.fit(X, y)


def test_search_train_scores_set_to_false():
    X = np.arange(6).reshape(6, -1)
    y = [0, 0, 0, 1, 1, 1]
    clf = LinearSVC(random_state=0)

    gs = dcv.GridSearchCV(clf, param_grid={"C": [0.1, 0.2]}, return_train_score=False)
    gs.fit(X, y)
    for key in gs.cv_results_:
        assert not key.endswith("train_score")

    if SK_VERSION >= packaging.version.parse("0.22.dev0"):
        gs = dcv.GridSearchCV(clf, param_grid={"C": [0.1, 0.2]})
        gs.fit(X, y)
        for key in gs.cv_results_:
            assert not key.endswith("train_score")


def test_multiple_metrics():
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_parameters_`` and
    # ``gs.best_index_``
    gs = dcv.GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={"min_samples_split": range(2, 403, 10)},
        scoring=scoring,
        cv=5,
        refit="AUC",
        return_train_score=True,
    )
    gs.fit(da_X, da_y)
    # some basic checks
    assert set(gs.scorer_) == {"AUC", "Accuracy"}
    cv_results = gs.cv_results_.keys()
    assert "split0_test_AUC" in cv_results
    assert "split0_train_AUC" in cv_results

    assert "split0_test_Accuracy" in cv_results
    assert "split0_test_Accuracy" in cv_results

    assert "mean_train_AUC" in cv_results
    assert "mean_train_Accuracy" in cv_results

    assert "std_train_AUC" in cv_results
    assert "std_train_Accuracy" in cv_results
