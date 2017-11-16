from __future__ import absolute_import, division, print_function

import os
import pickle
import warnings
from itertools import product
from multiprocessing import cpu_count

import pytest
import numpy as np
import pandas as pd

import dask
import dask.array as da
from dask.base import tokenize
from dask.callbacks import Callback
from dask.delayed import delayed
from dask.threaded import get as get_threading
from dask.utils import tmpdir

from sklearn.datasets import make_classification, load_iris
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.scorer import _passthrough_scorer
from sklearn.model_selection import (KFold,
                                     GroupKFold,
                                     StratifiedKFold,
                                     TimeSeriesSplit,
                                     ShuffleSplit,
                                     GroupShuffleSplit,
                                     StratifiedShuffleSplit,
                                     LeaveOneOut,
                                     LeavePOut,
                                     LeaveOneGroupOut,
                                     LeavePGroupsOut,
                                     PredefinedSplit,
                                     GridSearchCV)
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

import dask_searchcv as dcv
from dask_searchcv.model_selection import (compute_n_splits, check_cv,
                                           _normalize_n_jobs, _normalize_scheduler)
from dask_searchcv._compat import _HAS_MULTIPLE_METRICS
from dask_searchcv.methods import CVCache
from dask_searchcv.utils_test import (FailingClassifier, MockClassifier,
                                      ScalingTransformer, CheckXClassifier,
                                      ignore_warnings)

try:
    from distributed import Client
    from distributed.utils_test import cluster, loop
    has_distributed = True
except ImportError:
    loop = pytest.fixture(lambda: None)
    has_distributed = False


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
    pytest.importorskip('graphviz')

    X, y = make_classification(n_samples=100, n_classes=2, flip_y=.2,
                               random_state=0)
    clf = SVC(random_state=0)
    grid = {'C': [.1, .5, .9]}
    gs = dcv.GridSearchCV(clf, grid).fit(X, y)

    assert hasattr(gs, 'dask_graph_')

    with tmpdir() as d:
        gs.visualize(filename=os.path.join(d, 'mydask'))
        assert os.path.exists(os.path.join(d, 'mydask.png'))

    # Doesn't work if not fitted
    gs = dcv.GridSearchCV(clf, grid)
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


@pytest.mark.parametrize(['cls', 'has_shuffle'],
                         [(KFold, True),
                          (GroupKFold, False),
                          (StratifiedKFold, True),
                          (TimeSeriesSplit, False)])
def test_kfolds(cls, has_shuffle):
    assert tokenize(cls()) == tokenize(cls())
    assert tokenize(cls(n_splits=3)) != tokenize(cls(n_splits=4))
    if has_shuffle:
        assert (tokenize(cls(shuffle=True, random_state=0)) ==
                tokenize(cls(shuffle=True, random_state=0)))

        rs = np.random.RandomState(42)
        assert (tokenize(cls(shuffle=True, random_state=rs)) ==
                tokenize(cls(shuffle=True, random_state=rs)))

        assert (tokenize(cls(shuffle=True, random_state=0)) !=
                tokenize(cls(shuffle=True, random_state=2)))

        assert (tokenize(cls(shuffle=False, random_state=0)) ==
                tokenize(cls(shuffle=False, random_state=2)))

    cv = cls(n_splits=3)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == 3

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == 3


@pytest.mark.parametrize('cls', [ShuffleSplit, GroupShuffleSplit,
                                 StratifiedShuffleSplit])
def test_shuffle_split(cls):
    assert (tokenize(cls(n_splits=3, random_state=0)) ==
            tokenize(cls(n_splits=3, random_state=0)))

    assert (tokenize(cls(n_splits=3, random_state=0)) !=
            tokenize(cls(n_splits=3, random_state=2)))

    assert (tokenize(cls(n_splits=3, random_state=0)) !=
            tokenize(cls(n_splits=4, random_state=0)))

    cv = cls(n_splits=3)
    assert compute_n_splits(cv, np_X, np_y, np_groups) == 3

    with assert_dask_compute(False):
        assert compute_n_splits(cv, da_X, da_y, da_groups) == 3


@pytest.mark.parametrize('cvs', [(LeaveOneOut(),),
                                 (LeavePOut(2), LeavePOut(3))])
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


@pytest.mark.parametrize('cvs', [(LeaveOneGroupOut(),),
                                 (LeavePGroupsOut(2), LeavePGroupsOut(3))])
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
    cv1 = _CVIterableWrapper([np.array([True, False, True, False] * 5),
                              np.array([False, True, False, True] * 5)])
    cv2 = _CVIterableWrapper([np.array([True, False, True, False] * 5),
                              np.array([False, True, True, True] * 5)])
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
    for y in [np.array([0, 1, 0, 1, 0, 0, 1, 1, 1]),
              np.array([0, 1, 0, 1, 2, 1, 2, 0, 2])]:
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
        assert isinstance(check_cv(cv, y=dy, classifier=True),
                          _CVIterableWrapper)

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

    cv = GroupKFold()
    clf = SVC(random_state=0)
    grid = {'C': [1]}

    sol = SVC(C=1, random_state=0).fit(np_X, np_y).support_vectors_

    for X, y, groups in product([np_X, da_X, del_X],
                                [np_y, da_y, del_y],
                                [np_groups, da_groups, del_groups]):
        gs = dcv.GridSearchCV(clf, grid, cv=cv)

        with pytest.raises(ValueError) as exc:
            gs.fit(X, y)
        assert "parameter should not be None" in str(exc.value)

        gs.fit(X, y, groups=groups)
        np.testing.assert_allclose(sol, gs.best_estimator_.support_vectors_)


def test_pipeline_feature_union():
    iris = load_iris()
    X, y = iris.data, iris.target

    pca = PCA(random_state=0)
    kbest = SelectKBest()
    empty_union = FeatureUnion([('first', None), ('second', None)])
    empty_pipeline = Pipeline([('first', None), ('second', None)])
    scaling = Pipeline([('transform', ScalingTransformer())])
    svc = SVC(kernel='linear', random_state=0)

    pipe = Pipeline([('empty_pipeline', empty_pipeline),
                     ('scaling', scaling),
                     ('missing', None),
                     ('union', FeatureUnion([('pca', pca),
                                             ('missing', None),
                                             ('kbest', kbest),
                                             ('empty_union', empty_union)],
                                            transformer_weights={'pca': 0.5})),
                     ('svc', svc)])

    param_grid = dict(scaling__transform__factor=[1, 2],
                      union__pca__n_components=[1, 2, 3],
                      union__kbest__k=[1, 2],
                      svc__C=[0.1, 1, 10])

    gs = GridSearchCV(pipe, param_grid=param_grid)
    gs.fit(X, y)
    dgs = dcv.GridSearchCV(pipe, param_grid=param_grid, scheduler='sync')
    dgs.fit(X, y)

    # Check best params match
    assert gs.best_params_ == dgs.best_params_

    # Check PCA components match
    sk_pca = gs.best_estimator_.named_steps['union'].transformer_list[0][1]
    dk_pca = dgs.best_estimator_.named_steps['union'].transformer_list[0][1]
    np.testing.assert_allclose(sk_pca.components_, dk_pca.components_)

    # Check SelectKBest scores match
    sk_kbest = gs.best_estimator_.named_steps['union'].transformer_list[2][1]
    dk_kbest = dgs.best_estimator_.named_steps['union'].transformer_list[2][1]
    np.testing.assert_allclose(sk_kbest.scores_, dk_kbest.scores_)

    # Check SVC coefs match
    np.testing.assert_allclose(gs.best_estimator_.named_steps['svc'].coef_,
                               dgs.best_estimator_.named_steps['svc'].coef_)


def test_pipeline_sub_estimators():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaling = Pipeline([('transform', ScalingTransformer())])

    pipe = Pipeline([('setup', None),
                     ('missing', None),
                     ('scaling', scaling),
                     ('svc', SVC(kernel='linear', random_state=0))])

    param_grid = [{'svc__C': [0.1, 0.1]},  # Duplicates to test culling
                  {'setup': [None],
                   'svc__C': [0.1, 1, 10],
                   'scaling': [ScalingTransformer(), None]},
                  {'setup': [SelectKBest()],
                   'setup__k': [1, 2],
                   'svc': [SVC(kernel='linear', random_state=0, C=0.1),
                           SVC(kernel='linear', random_state=0, C=1),
                           SVC(kernel='linear', random_state=0, C=10)]}]

    gs = GridSearchCV(pipe, param_grid=param_grid, return_train_score=True)
    gs.fit(X, y)
    dgs = dcv.GridSearchCV(pipe, param_grid=param_grid, scheduler='sync',
                           return_train_score=True)
    dgs.fit(X, y)

    # Check best params match
    assert gs.best_params_ == dgs.best_params_

    # Check cv results match
    res = pd.DataFrame(dgs.cv_results_)
    sol = pd.DataFrame(gs.cv_results_)
    assert res.columns.equals(sol.columns)
    skip = ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
    res = res.drop(skip, axis=1)
    sol = sol.drop(skip, axis=1)
    assert res.equals(sol)

    # Check SVC coefs match
    np.testing.assert_allclose(gs.best_estimator_.named_steps['svc'].coef_,
                               dgs.best_estimator_.named_steps['svc'].coef_)


def check_scores_all_nan(gs, bad_param, score_key='score'):
    bad_param = 'param_' + bad_param
    n_candidates = len(gs.cv_results_['params'])
    keys = ['split{}_test_{}'.format(s, score_key)
            for s in range(gs.n_splits_)]
    assert all(np.isnan([gs.cv_results_[key][cand_i]
                         for key in keys]).all()
               for cand_i in range(n_candidates)
               if gs.cv_results_[bad_param][cand_i] ==
               FailingClassifier.FAILING_PARAMETER)


@pytest.mark.parametrize('weights',
                         [None, (None, {'tr0': 2, 'tr2': 3}, {'tr0': 2, 'tr2': 4})])
def test_feature_union(weights):
    X = np.ones((10, 5))
    y = np.zeros(10)

    union = FeatureUnion([('tr0', ScalingTransformer()),
                          ('tr1', ScalingTransformer()),
                          ('tr2', ScalingTransformer())])

    factors = [(2, 3, 5), (2, 4, 5), (2, 4, 6),
               (2, 4, None), (None, None, None)]
    params, sols, grid = [], [], []
    for constants, w in product(factors, weights or [None]):
        p = {}
        for n, c in enumerate(constants):
            if c is None:
                p['tr%d' % n] = None
            elif n == 3:  # 3rd is always an estimator
                p['tr%d' % n] = ScalingTransformer(c)
            else:
                p['tr%d__factor' % n] = c
        sol = union.set_params(transformer_weights=w, **p).transform(X)
        sols.append(sol)
        if w is not None:
            p['transformer_weights'] = w
        params.append(p)
        p2 = {'union__' + k: [v] for k, v in p.items()}
        p2['est'] = [CheckXClassifier(sol[0])]
        grid.append(p2)

    # Need to recreate the union after setting estimators to `None` above
    union = FeatureUnion([('tr0', ScalingTransformer()),
                          ('tr1', ScalingTransformer()),
                          ('tr2', ScalingTransformer())])

    pipe = Pipeline([('union', union), ('est', CheckXClassifier())])
    gs = dcv.GridSearchCV(pipe, grid, refit=False, cv=2)

    with warnings.catch_warnings(record=True):
        gs.fit(X, y)


@ignore_warnings
def test_feature_union_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('union', FeatureUnion([('good', MockClassifier()),
                                             ('bad', FailingClassifier())],
                                            transformer_weights={'bad': 0.5})),
                     ('clf', MockClassifier())])

    grid = {'union__bad__parameter': [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False, scoring=None)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float('nan')
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)
    check_scores_all_nan(gs, 'union__bad__parameter')


@ignore_warnings
@pytest.mark.skipif(not _HAS_MULTIPLE_METRICS, reason="Added in 0.19.0")
def test_feature_union_fit_failure_multiple_metrics():
    scoring = {"score_1": _passthrough_scorer, "score_2": _passthrough_scorer}
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('union', FeatureUnion([('good', MockClassifier()),
                                             ('bad', FailingClassifier())],
                                            transformer_weights={'bad': 0.5})),
                     ('clf', MockClassifier())])

    grid = {'union__bad__parameter': [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False, scoring=scoring)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float('nan')
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    for key in scoring:
        check_scores_all_nan(gs, 'union__bad__parameter', score_key=key)


@ignore_warnings
def test_pipeline_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('bad', FailingClassifier()),
                     ('good1', MockClassifier()),
                     ('good2', MockClassifier())])

    grid = {'bad__parameter': [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float('nan')
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    check_scores_all_nan(gs, 'bad__parameter')


def test_pipeline_raises():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('step1', MockClassifier()),
                     ('step2', MockClassifier())])

    grid = {'step3__parameter': [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False)
    with pytest.raises(ValueError):
        gs.fit(X, y)

    grid = {'steps': [[('one', MockClassifier()), ('two', MockClassifier())]]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False)
    with pytest.raises(NotImplementedError):
        gs.fit(X, y)


def test_feature_union_raises():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    union = FeatureUnion([('tr0', MockClassifier()),
                          ('tr1', MockClassifier())])
    pipe = Pipeline([('union', union), ('est', MockClassifier())])

    grid = {'union__tr2__parameter': [0, 1, 2]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False)
    with pytest.raises(ValueError):
        gs.fit(X, y)

    grid = {'union__transformer_list': [[('one', MockClassifier())]]}
    gs = dcv.GridSearchCV(pipe, grid, refit=False)
    with pytest.raises(NotImplementedError):
        gs.fit(X, y)


def test_bad_error_score():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    gs = dcv.GridSearchCV(MockClassifier(), {'foo_param': [0, 1, 2]},
                          error_score='badparam')

    with pytest.raises(ValueError):
        gs.fit(X, y)


class CountTakes(np.ndarray):
    count = 0

    def take(self, *args, **kwargs):
        self.count += 1
        return super(CountTakes, self).take(*args, **kwargs)


def test_cache_cv():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    X2 = X.view(CountTakes)
    gs = dcv.GridSearchCV(MockClassifier(), {'foo_param': [0, 1, 2]},
                          cv=3, cache_cv=False, scheduler='sync')
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
    assert all((cache2.splits[i][j] == cache.splits[i][j]).all()
               for i in range(2) for j in range(2))


def test_normalize_n_jobs():
    assert _normalize_n_jobs(-1) is None
    assert _normalize_n_jobs(-2) == cpu_count() - 1
    with pytest.raises(TypeError):
        _normalize_n_jobs('not an integer')


@pytest.mark.parametrize('scheduler,n_jobs,get',
                         [(None, 4, get_threading),
                          ('threading', 4, get_threading),
                          ('threaded', 4, get_threading),
                          ('threading', 1, dask.get),
                          ('sequential', 4, dask.get),
                          ('synchronous', 4, dask.get),
                          ('sync', 4, dask.get),
                          ('multiprocessing', 4, None),
                          (dask.get, 4, dask.get)])
def test_scheduler_param(scheduler, n_jobs, get):
    if scheduler == 'multiprocessing':
        mp = pytest.importorskip('dask.multiprocessing')
        get = mp.get

    assert _normalize_scheduler(scheduler, n_jobs) is get

    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    gs = dcv.GridSearchCV(MockClassifier(), {'foo_param': [0, 1, 2]}, cv=3,
                          scheduler=scheduler, n_jobs=n_jobs)
    gs.fit(X, y)


@pytest.mark.skipif('not has_distributed')
def test_scheduler_param_distributed(loop):
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop, set_as_default=False) as client:
            gs = dcv.GridSearchCV(MockClassifier(), {'foo_param': [0, 1, 2]},
                                  cv=3, scheduler=client)
            gs.fit(X, y)


def test_scheduler_param_bad():
    with pytest.raises(ValueError):
        _normalize_scheduler('threeding', 4)


@pytest.mark.skipif(not _HAS_MULTIPLE_METRICS, reason="Added in 0.19.0")
def test_cv_multiplemetrics():
    X, y = make_classification(random_state=0)

    param_grid = {'max_depth': [1, 5]}
    a = dcv.GridSearchCV(RandomForestClassifier(), param_grid, refit='score1',
                         scoring={'score1': 'accuracy', 'score2': 'accuracy'})
    b = GridSearchCV(RandomForestClassifier(), param_grid, refit='score1',
                     scoring={'score1': 'accuracy', 'score2': 'accuracy'})
    a.fit(X, y)
    b.fit(X, y)

    assert a.best_score_ > 0
    assert isinstance(a.best_index_, type(b.best_index_))
    assert isinstance(a.best_params_, type(b.best_params_))


@pytest.mark.skipif(not _HAS_MULTIPLE_METRICS, reason="Added in 0.19.0")
def test_cv_multiplemetrics_requires_refit_metric():
    X, y = make_classification(random_state=0)

    param_grid = {'max_depth': [1, 5]}
    a = dcv.GridSearchCV(RandomForestClassifier(), param_grid, refit=True,
                         scoring={'score1': 'accuracy', 'score2': 'accuracy'})

    with pytest.raises(ValueError):
        a.fit(X, y)


@pytest.mark.skipif(not _HAS_MULTIPLE_METRICS, reason="Added in 0.19.0")
def test_cv_multiplemetrics_no_refit():
    X, y = make_classification(random_state=0)

    param_grid = {'max_depth': [1, 5]}
    a = dcv.GridSearchCV(RandomForestClassifier(), param_grid, refit=False,
                         scoring={'score1': 'accuracy', 'score2': 'accuracy'})
    b = GridSearchCV(RandomForestClassifier(), param_grid, refit=False,
                     scoring={'score1': 'accuracy', 'score2': 'accuracy'})
    assert hasattr(a, 'best_index_') is hasattr(b, 'best_index_')
    assert hasattr(a, 'best_estimator_') is hasattr(b, 'best_estimator_')
    assert hasattr(a, 'best_score_') is hasattr(b, 'best_score_')
