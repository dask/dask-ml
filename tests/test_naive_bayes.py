from dask.array.utils import assert_eq
from daskml.datasets import make_classification
from daskml.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB as GaussianNB_

X, y = make_classification(chunks=50)
X_ = X.compute()
y_ = y.compute()


def test_smoke():
    nb = GaussianNB()
    nb_ = GaussianNB_()
    nb.fit(X, y)
    nb_.fit(X.compute(), y.compute())

    assert_eq(nb.class_prior_.compute(), nb_.class_prior_)
    assert_eq(nb.class_count_.compute(), nb_.class_count_)
    assert_eq(nb.theta_.compute(), nb_.theta_)
    assert_eq(nb.sigma_.compute(), nb_.sigma_)

    assert_eq(nb.predict_proba(X).compute(), nb_.predict_proba(X_))
    assert_eq(nb.predict(X).compute(), nb_.predict(X_))
    assert_eq(nb.predict_log_proba(X).compute(), nb_.predict_log_proba(X_))
