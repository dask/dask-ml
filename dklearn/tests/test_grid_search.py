from __future__ import print_function, absolute_import, division

import numpy as np
import numpy.testing.utils as tm
import dask.array as da
from scipy import sparse
from scipy.stats import expon
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans

from dklearn import Chained
from dklearn.grid_search import GridSearchCV, RandomizedSearchCV


# Several of these tests were copied (with modification) from the equivalent
# scikit-learn testing code. The scikit-learn license has been included at
# dklearn/SCIKIT_LEARN_LICENSE.txt.


class MockClassifier(BaseEstimator):
    """Dummy classifier to test the cross-validation"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert len(X) == len(Y)
        return self

    def predict(self, T):
        return T.sum(axis=1)

    predict_proba = predict
    decision_function = predict
    transform = predict

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score

    def get_params(self, deep=False):
        return {'foo_param': self.foo_param}

    def set_params(self, **params):
        self.foo_param = params['foo_param']
        return self


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""
    @property
    def score(self):
        raise AttributeError


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def test_grid_search_numpy_inputs():
    # Test that the best estimator contains the right value for foo_param
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {'foo_param': [1, 2, 3]})
    # make sure it selects the smallest parameter in case of ties
    grid_search.fit(X, y)
    assert grid_search.best_estimator_.foo_param == 2

    for i, foo_i in enumerate([1, 2, 3]):
        assert grid_search.grid_scores_[i][0] == {'foo_param': foo_i}


def test_grid_search_dask_inputs():
    # Test that the best estimator contains the right value for foo_param
    dX = da.from_array(X, chunks=2)
    dy = da.from_array(y, chunks=2)
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {'foo_param': [1, 2, 3]})
    # make sure it selects the smallest parameter in case of ties
    grid_search.fit(dX, dy)
    assert grid_search.best_estimator_.foo_param == 2

    for i, foo_i in enumerate([1, 2, 3]):
        assert grid_search.grid_scores_[i][0] == {'foo_param': foo_i}

    y_pred = grid_search.predict(dX)
    assert isinstance(y_pred, da.Array)
    tm.assert_array_equal(y_pred, X.sum(axis=1))

    y_pred = grid_search.predict(X)
    assert isinstance(y_pred, np.ndarray)
    tm.assert_array_equal(y_pred, X.sum(axis=1))


def test_grid_search_dask_inputs_dk_est():
    X, y = make_classification(n_samples=1000, n_features=100, random_state=0)
    dX = da.from_array(X, chunks=100)
    dy = da.from_array(y, chunks=100)
    grid = {'alpha': [0.1, 0.01, 0.0001]}

    clf = SGDClassifier()
    d_clf = Chained(SGDClassifier())
    grid_search = GridSearchCV(clf, grid)
    d_grid_search = GridSearchCV(d_clf, grid, fit_params={'classes': [0, 1]})

    grid_search.fit(X, y)
    d_grid_search.fit(dX, dy)
    assert (d_grid_search.best_estimator_.alpha ==
            grid_search.best_estimator_.alpha)


def test_grid_search_no_refit():
    # Test that grid search can be used for model selection only
    clf = MockClassifier()
    grid_search = GridSearchCV(clf, {'foo_param': [1, 2, 3]}, refit=False)
    grid_search.fit(X, y)
    assert hasattr(grid_search, "best_params_")
    assert not hasattr(grid_search, "best_estimator_")


def test_grid_search_iid():
    # test the iid parameter
    # noise-free simple 2d-data
    X, y = make_blobs(centers=[[0, 0], [1, 0], [0, 1], [1, 1]], random_state=0,
                      cluster_std=0.1, shuffle=False, n_samples=80)
    # split dataset into two folds that are not iid
    # first one contains data of all 4 blobs, second only from two.
    mask = np.ones(X.shape[0], dtype=np.bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0
    # this leads to perfect classification on one fold and a score of 1/3 on
    # the other
    svm = SVC(kernel='linear')
    # create "cv" for splits
    cv = [[mask, ~mask], [~mask, mask]]
    # once with iid=True (default)
    grid_search = GridSearchCV(svm, param_grid={'C': [1, 10]}, cv=cv)
    grid_search.fit(X, y)
    first = grid_search.grid_scores_[0]
    tm.assert_equal(first.parameters['C'], 1)
    tm.assert_array_almost_equal(first.cv_validation_scores, [1, 1. / 3.])
    # for first split, 1/4 of dataset is in test, for second 3/4.
    # take weighted average
    tm.assert_almost_equal(first.mean_validation_score,
                           1 * 1. / 4. + 1. / 3. * 3. / 4.)

    # once with iid=False
    grid_search = GridSearchCV(svm, param_grid={'C': [1, 10]}, cv=cv,
                               iid=False)
    grid_search.fit(X, y)
    first = grid_search.grid_scores_[0]
    assert first.parameters['C'] == 1
    # scores are the same as above
    tm.assert_array_almost_equal(first.cv_validation_scores, [1, 1. / 3.])
    # averaged score is just mean of scores
    tm.assert_almost_equal(first.mean_validation_score,
                           np.mean(first.cv_validation_scores))


def test_grid_search_one_grid_point():
    X, y = make_classification(n_samples=200, n_features=100, random_state=0)
    param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

    clf = SVC()
    cv = GridSearchCV(clf, param_dict)
    cv.fit(X, y)

    clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
    clf.fit(X, y)

    tm.assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)


def test_grid_search_sparse():
    # Test that grid search works with both dense and sparse matrices
    X, y = make_classification(n_samples=200, n_features=100, random_state=0)

    cv = GridSearchCV(LinearSVC(), {'C': [0.1, 1.0]})
    cv.fit(X[:180], y[:180])
    y_pred = cv.predict(X[180:])
    C = cv.best_estimator_.C

    X = sparse.csr_matrix(X)
    cv.fit(X[:180], y[:180])
    y_pred2 = cv.predict(X[180:])
    C2 = cv.best_estimator_.C

    assert np.mean(y_pred == y_pred2) >= .9
    assert C == C2


def test_unsupervised_grid_search():
    # test grid-search with unsupervised estimator
    X, y = make_blobs(random_state=0)
    km = KMeans(random_state=0)
    grid_search = GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]),
                               scoring='adjusted_rand_score')
    grid_search.fit(X, y)
    # ARI can find the right number :)
    assert grid_search.best_params_["n_clusters"] == 3

    # Now without a score, and without y
    grid_search = GridSearchCV(km, param_grid=dict(n_clusters=[2, 3, 4]))
    grid_search.fit(X)
    assert grid_search.best_params_["n_clusters"] == 4


def test_randomized_search_grid_scores():
    # Make a dataset with a lot of noise to get various kind of prediction
    # errors across CV folds and parameter settings
    X, y = make_classification(n_samples=200, n_features=100, n_informative=3,
                               random_state=0)

    # XXX: as of today (scipy 0.12) it's not possible to set the random seed
    # of scipy.stats distributions: the assertions in this test should thus
    # not depend on the randomization
    params = dict(C=expon(scale=10),
                  gamma=expon(scale=0.1))
    n_cv_iter = 3
    n_search_iter = 30
    search = RandomizedSearchCV(SVC(), n_iter=n_search_iter, cv=n_cv_iter,
                                param_distributions=params, iid=False)
    search.fit(X, y)
    assert len(search.grid_scores_) == n_search_iter

    # Check consistency of the structure of each cv_score item
    for cv_score in search.grid_scores_:
        assert len(cv_score.cv_validation_scores) == n_cv_iter
        # Because we set iid to False, the mean_validation score is the
        # mean of the fold mean scores instead of the aggregate sample-wise
        # mean score
        tm.assert_almost_equal(np.mean(cv_score.cv_validation_scores),
                               cv_score.mean_validation_score)
        assert (list(sorted(cv_score.parameters.keys())) ==
                list(sorted(params.keys())))

    # Check the consistency with the best_score_ and best_params_ attributes
    sorted_grid_scores = list(sorted(search.grid_scores_,
                              key=lambda x: x.mean_validation_score))
    best_score = sorted_grid_scores[-1].mean_validation_score
    assert search.best_score_ == best_score

    tied_best_params = [s.parameters for s in sorted_grid_scores
                        if s.mean_validation_score == best_score]
    assert search.best_params_ in tied_best_params


def test_grid_search_score_consistency():
    # test that correct scores are used
    clf = LinearSVC(random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [.1, 1, 10]
    for score in ['f1', 'roc_auc']:
        grid_search = GridSearchCV(clf, {'C': Cs}, scoring=score)
        grid_search.fit(X, y)
        cv = StratifiedKFold(n_folds=3, y=y)
        for C, scores in zip(Cs, grid_search.grid_scores_):
            clf.set_params(C=C)
            scores = scores[2]  # get the separate runs from grid scores
            i = 0
            for train, test in cv:
                clf.fit(X[train], y[train])
                if score == "f1":
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == "roc_auc":
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                tm.assert_almost_equal(correct_score, scores[i])
                i += 1
