from __future__ import absolute_import, division, print_function

import os
from itertools import product

import pytest
import numpy as np

import dask
import dask.array as da
from dask.base import tokenize
from dask.callbacks import Callback
from dask.delayed import delayed
from dask.utils import tmpdir

from sklearn.datasets import make_classification, load_iris
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.feature_selection import SelectKBest
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

from dklearn import DaskGridSearchCV
from dklearn._builder import compute_n_splits, check_cv
from dklearn.utils_test import (FailingClassifier, MockClassifier,
                                ignore_warnings)


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
    gs = DaskGridSearchCV(clf, grid).fit(X, y)

    assert hasattr(gs, 'dask_graph_')

    with tmpdir() as d:
        gs.visualize(filename=os.path.join(d, 'mydask'))
        assert os.path.exists(os.path.join(d, 'mydask.png'))

    # Doesn't work if not fitted
    gs = DaskGridSearchCV(clf, grid)
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
        gs = DaskGridSearchCV(clf, grid, cv=cv)

        with pytest.raises(ValueError) as exc:
            gs.fit(X, y)
        assert "The groups parameter should not be None" in str(exc.value)

        gs.fit(X, y, groups=groups)
        np.testing.assert_allclose(sol, gs.best_estimator_.support_vectors_)


def test_pipeline_feature_union():
    iris = load_iris()
    X, y = iris.data, iris.target

    pca = PCA(random_state=0)
    kbest = SelectKBest()
    empty_union = FeatureUnion([('first', None), ('second', None)])
    empty_pipeline = Pipeline([('first', None), ('second', None)])
    svc = SVC(kernel='linear', random_state=0)

    pipe = Pipeline([('empty_pipeline', empty_pipeline),
                     ('missing', None),
                     ('union', FeatureUnion([('pca', pca),
                                             ('missing', None),
                                             ('kbest', kbest),
                                             ('empty_union', empty_union)],
                                            transformer_weights={'pca': 0.5})),
                     ('svc', svc)])

    param_grid = dict(union__pca__n_components=[1, 2, 3],
                      union__kbest__k=[1, 2],
                      svc__C=[0.1, 1, 10])

    gs = GridSearchCV(pipe, param_grid=param_grid)
    gs.fit(X, y)
    dgs = DaskGridSearchCV(pipe, param_grid=param_grid, get=dask.get)
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


def check_scores_all_nan(gs, bad_param):
    bad_param = 'param_' + bad_param
    n_candidates = len(gs.cv_results_['params'])
    assert all(np.isnan([gs.cv_results_['split%d_test_score' % s][cand_i]
                        for s in range(gs.n_splits_)]).all()
               for cand_i in range(n_candidates)
               if gs.cv_results_[bad_param][cand_i] ==
               FailingClassifier.FAILING_PARAMETER)


@ignore_warnings
def test_feature_union_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('union', FeatureUnion([('good', MockClassifier()),
                                             ('bad', FailingClassifier())],
                                            transformer_weights={'bad': 0.5})),
                     ('clf', MockClassifier())])

    grid = {'union__bad__parameter': [0, 1, 2]}
    gs = DaskGridSearchCV(pipe, grid, refit=False)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float('nan')
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    check_scores_all_nan(gs, 'union__bad__parameter')


@ignore_warnings
def test_pipeline_fit_failure():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)

    pipe = Pipeline([('bad', FailingClassifier()),
                     ('good1', MockClassifier()),
                     ('good2', MockClassifier())])

    grid = {'bad__parameter': [0, 1, 2]}
    gs = DaskGridSearchCV(pipe, grid, refit=False)

    # Check that failure raises if error_score is `'raise'`
    with pytest.raises(ValueError):
        gs.fit(X, y)

    # Check that grid scores were set to error_score on failure
    gs.error_score = float('nan')
    with pytest.warns(FitFailedWarning):
        gs.fit(X, y)

    check_scores_all_nan(gs, 'bad__parameter')


def test_bad_error_score():
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    gs = DaskGridSearchCV(MockClassifier(), {'foo_param': [0, 1, 2]},
                          error_score='badparam')

    with pytest.raises(ValueError):
        gs.fit(X, y)
