import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder

from dask_ml.datasets import make_classification
from dask_ml.utils import assert_eq_ar, assert_estimator_equal
from dask_ml.wrappers import ParallelPostFit


def test_it_works():
    clf = ParallelPostFit(GradientBoostingClassifier())

    X, y = make_classification(n_samples=1000, chunks=100)
    X_, y_ = dask.compute(X, y)
    clf.fit(X_, y_)

    assert isinstance(clf.predict(X), da.Array)
    assert isinstance(clf.predict_proba(X), da.Array)

    result = clf.score(X, y)
    expected = clf.estimator.score(X_, y_)
    assert result == expected


def test_no_method_raises():
    clf = ParallelPostFit(LinearRegression())
    X, y = make_classification(chunks=50)
    clf.fit(X, y)

    with pytest.raises(AttributeError) as m:
        clf.predict_proba(X)

    assert m.match("The wrapped estimator (.|\n)* 'predict_proba' method.")


def test_laziness():
    clf = ParallelPostFit(LinearRegression())
    X, y = make_classification(chunks=50)
    clf.fit(X, y)

    x = clf.score(X, y, compute=False)
    assert dask.is_dask_collection(x)
    assert 0 < x.compute() < 1


def test_predict_meta_override():
    X = pd.DataFrame({"c_0": [1, 2, 3, 4]})
    y = np.array([1, 2, 3, 4])

    base = CategoricalNB()
    base.fit(pd.DataFrame(X), y)

    dd_X = dd.from_pandas(X, npartitions=2)
    # Success when providing meta over-ride
    wrap = ParallelPostFit(base, predict_meta=np.array([1]))
    result = wrap.predict(dd_X)
    expected = base.predict(X)
    assert_eq_ar(result, expected)


def test_predict_proba_meta_override():
    X = pd.DataFrame({"c_0": [1, 2, 3, 4]})
    y = np.array([1, 2, 3, 4])

    base = CategoricalNB()
    base.fit(pd.DataFrame(X), y)

    dd_X = dd.from_pandas(X, npartitions=2)

    # Success when providing meta over-ride
    wrap = ParallelPostFit(base, predict_proba_meta=np.array([[0.0, 0.1, 0.8, 0.1]]))
    result = wrap.predict_proba(dd_X)
    expected = base.predict_proba(X)
    assert_eq_ar(result, expected)


def test_transform_meta_override():
    X = pd.DataFrame({"cat_s": ["a", "b", "c", "d"]})
    dd_X = dd.from_pandas(X, npartitions=2)

    base = OneHotEncoder(sparse_output=False)
    base.fit(pd.DataFrame(X))

    # Failure when not proving transform_meta
    # because of value dependent model
    wrap = ParallelPostFit(base)
    with pytest.raises(ValueError):
        wrap.transform(dd_X)

    wrap = ParallelPostFit(
        base, transform_meta=np.array([[0, 0, 0, 0]], dtype=np.float64)
    )
    result = wrap.transform(dd_X)
    expected = base.transform(X)
    assert_eq_ar(result, expected)


def test_predict_correct_output_dtype():
    X, y = make_classification(chunks=100)
    X_ddf = dd.from_dask_array(X)

    base = LinearRegression(n_jobs=1)
    base.fit(X, y)

    wrap = ParallelPostFit(base)

    base_output = base.predict(X_ddf.compute())
    wrap_output = wrap.predict(X_ddf)

    assert wrap_output.dtype == base_output.dtype


@pytest.mark.parametrize("kind", ["numpy", "dask.dataframe", "dask.array"])
def test_predict(kind):
    X, y = make_classification(chunks=100)

    if kind == "numpy":
        X, y = dask.compute(X, y)
    elif kind == "dask.dataframe":
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

    base = LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs")
    wrap = ParallelPostFit(
        LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs"),
    )

    base.fit(*dask.compute(X, y))
    wrap.fit(*dask.compute(X, y))

    assert_estimator_equal(wrap.estimator, base)

    result = wrap.predict(X)
    expected = base.predict(X)
    assert_eq_ar(result, expected)

    result = wrap.predict_proba(X)
    expected = base.predict_proba(X)
    assert_eq_ar(result, expected)

    result = wrap.predict_log_proba(X)
    expected = base.predict_log_proba(X)
    assert_eq_ar(result, expected)


@pytest.mark.parametrize("kind", ["numpy", "dask.dataframe", "dask.array"])
def test_transform(kind):
    X, y = make_classification(chunks=100)

    if kind == "numpy":
        X, y = dask.compute(X, y)
    elif kind == "dask.dataframe":
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

    base = PCA(random_state=0)
    wrap = ParallelPostFit(PCA(random_state=0))

    base.fit(*dask.compute(X, y))
    wrap.fit(*dask.compute(X, y))

    assert_estimator_equal(wrap.estimator, base, exclude="n_features_")

    result = base.transform(*dask.compute(X))
    expected = wrap.transform(X)
    assert_eq_ar(result, expected)


def test_multiclass():
    X, y = sklearn.datasets.make_classification(n_classes=3, n_informative=4)
    X = da.from_array(X, chunks=50)
    y = da.from_array(y, chunks=50)

    clf = ParallelPostFit(LogisticRegression(random_state=0, n_jobs=1, solver="lbfgs"))

    clf.fit(*dask.compute(X, y))
    result = clf.predict(X)
    expected = clf.estimator.predict(X)

    assert isinstance(result, da.Array)
    assert_eq_ar(result, expected)

    result = clf.predict_proba(X)
    expected = clf.estimator.predict_proba(X)

    assert isinstance(result, da.Array)
    assert_eq_ar(result, expected)

    result = clf.predict_log_proba(X)
    expected = clf.estimator.predict_log_proba(X)
    assert_eq_ar(result, expected)


def test_auto_rechunk():
    clf = ParallelPostFit(GradientBoostingClassifier())
    X, y = make_classification(n_samples=1000, n_features=20, chunks=100)
    X = X.rechunk({0: 100, 1: 10})
    clf.fit(X, y)

    assert clf.predict(X).compute().shape == (1000,)
    assert clf.predict_proba(X).compute().shape == (1000, 2)
    assert clf.score(X, y) == clf.score(X.compute(), y.compute())

    X, y = make_classification(n_samples=1000, n_features=20, chunks=100)
    X = X.rechunk({0: 100, 1: 10})
    X._chunks = (tuple(np.nan for _ in X.chunks[0]), X.chunks[1])
    clf.predict(X)


def test_sparse_inputs():
    X = csr_matrix((3, 4))
    y = np.asarray([0, 0, 1], dtype=np.int32)

    base = SGDClassifier(tol=1e-3)
    base = base.fit(X, y)

    wrap = ParallelPostFit(base)
    X_da = da.from_array(X, chunks=(1, 4))

    result = wrap.predict(X_da).compute()
    expected = base.predict(X)

    assert_eq_ar(result, expected)


def test_warning_on_dask_array_without_array_function():
    X, y = make_classification(n_samples=10, n_features=2, chunks=10)
    clf = ParallelPostFit(GradientBoostingClassifier())
    clf = clf.fit(X, y)

    class FakeArray:
        def __init__(self, value):
            self.value = value

        @property
        def ndim(self):
            return self.value.ndim

        @property
        def len(self):
            return self.value.len

        @property
        def dtype(self):
            return self.value.dtype

        @property
        def shape(self):
            return self.value.shape

    ar = FakeArray(np.zeros(shape=(2, 2)))
    fake_dask_ar = da.from_array(ar)
    fake_dask_ar._meta = FakeArray(np.zeros(shape=(0, 0)))

    with pytest.warns(
        UserWarning, match="provide explicit `predict_meta` to the dask_ml.wrapper"
    ):
        clf.predict(fake_dask_ar)

    with pytest.warns(
        UserWarning,
        match="provide explicit `predict_proba_meta` to the dask_ml.wrapper",
    ):
        clf.predict_proba(fake_dask_ar)
