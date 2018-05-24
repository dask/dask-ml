from sklearn.svm import SVC
from scipy import stats

import dask_ml.model_selection as dms


def test_search_basic(xy_classification):
    X, y = xy_classification
    param_grid = {'class_weight': [None, 'balanced']}

    a = dms.GridSearchCV(SVC(kernel='rbf', gamma=0.1), param_grid)
    a.fit(X, y)

    param_dist = {'C': stats.uniform}
    b = dms.RandomizedSearchCV(SVC(kernel='rbf', gamma=0.1), param_dist)
    b.fit(X, y)
