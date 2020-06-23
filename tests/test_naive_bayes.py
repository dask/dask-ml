from dask.array.utils import assert_eq
from sklearn import naive_bayes as nb_

from dask_ml import naive_bayes as nb
from dask_ml.datasets import make_classification

X, y = make_classification(chunks=50)


def test_smoke():
    a = nb.GaussianNB()
    b = nb_.GaussianNB()
    a.fit(X, y)
    X_ = X.compute()
    y_ = y.compute()
    b.fit(X_, y_)

    assert_eq(a.class_prior_.compute(), b.class_prior_)
    assert_eq(a.class_count_.compute(), b.class_count_)
    assert_eq(a.theta_.compute(), b.theta_)
    assert_eq(a.sigma_.compute(), b.sigma_)

    assert_eq(a.predict_proba(X).compute(), b.predict_proba(X_))
    assert_eq(a.predict(X).compute(), b.predict(X_))
    assert_eq(a.predict_log_proba(X).compute(), b.predict_log_proba(X_))
