import dask
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import dask_ml  # noqa


def test_normalize_estimator():
    m1 = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    m2 = sklearn.linear_model.LogisticRegression(solver="lbfgs")

    assert dask.base.tokenize(m1) == dask.base.tokenize(m2)
    m1.fit(*sklearn.datasets.make_classification())
    m2.fit(*sklearn.datasets.make_classification())

    assert dask.base.tokenize(m1) != dask.base.tokenize(m2)


def test_normalize_estimator_cv():
    param_grid = {"C": [0.01]}
    a = sklearn.linear_model.LogisticRegression(random_state=0, solver="lbfgs")
    m1 = sklearn.model_selection.GridSearchCV(a, param_grid, cv=3)
    m2 = sklearn.model_selection.GridSearchCV(a, param_grid, cv=3)

    assert dask.base.tokenize(m1) == dask.base.tokenize(m2)
    X, y = sklearn.datasets.make_classification()
    m1.fit(X, y)
    m2.fit(X, y)

    assert dask.base.tokenize(m1) == dask.base.tokenize(m2)
