from dask.array.utils import assert_eq
from daskml.datasets import make_classification
from daskml import naive_bayes as nb
from sklearn import naive_bayes as nb_

X, y = make_classification(chunks=50)
X_ = X.compute()
y_ = y.compute()


def test_smoke():
    a = nb.GaussianNB()
    b = nb_.GaussianNB()
    a.fit(X, y)
    b.fit(X.compute(), y.compute())

    assert_eq(a.class_prior_.compute(), b.class_prior_)
    assert_eq(a.class_count_.compute(), b.class_count_)
    assert_eq(a.theta_.compute(), b.theta_)
    assert_eq(a.sigma_.compute(), b.sigma_)

    assert_eq(a.predict_proba(X).compute(), b.predict_proba(X_))
    assert_eq(a.predict(X).compute(), b.predict(X_))
    assert_eq(a.predict_log_proba(X).compute(), b.predict_log_proba(X_))


class TestBigMultinomialNB:
    def test_basic(self, single_chunk_count_classification):
        X, y = single_chunk_count_classification
        a = nb.BigMultinomialNB(classes=[0, 1])
        b = nb_.MultinomialNB()
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_eq(a.coef_, b.coef_)


class TestBigBernoulliNB:
    def test_basic(self, single_chunk_binary_classification):
        X, y = single_chunk_binary_classification
        a = nb.BigBernoulliNB(classes=[0, 1])
        b = nb_.BernoulliNB()
        a.fit(X, y)
        b.partial_fit(X, y, classes=[0, 1])
        assert_eq(a.coef_, b.coef_)
