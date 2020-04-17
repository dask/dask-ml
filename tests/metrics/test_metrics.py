import dask
import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import sklearn
import sklearn.linear_model
import sklearn.metrics
from dask.array.utils import assert_eq

import dask_ml.metrics
import dask_ml.wrappers


def test_pairwise_distances(X_blobs):
    centers = X_blobs[::100].compute()
    result = dask_ml.metrics.pairwise_distances(X_blobs, centers)
    expected = sklearn.metrics.pairwise_distances(X_blobs.compute(), centers)
    assert_eq(result, expected, atol=1e-4)


def test_pairwise_distances_argmin_min(X_blobs):
    centers = X_blobs[::100].compute()

    # X_blobs has 500 rows per block.
    # Ensure 500 rows in the scikit-learn version too.
    working_memory = float(80 * 500) / 2 ** 20

    ctx = sklearn.config_context(working_memory=working_memory)

    with ctx:
        a_, b_ = sklearn.metrics.pairwise_distances_argmin_min(
            X_blobs.compute(), centers
        )
        a, b = dask_ml.metrics.pairwise_distances_argmin_min(X_blobs, centers)
        a, b = dask.compute(a, b)

    npt.assert_array_equal(a, a_)
    npt.assert_array_equal(b, b_)


def test_euclidean_distances():
    X = da.random.uniform(size=(100, 4), chunks=50)
    Y = da.random.uniform(size=(100, 4), chunks=50)
    a = dask_ml.metrics.euclidean_distances(X, Y)
    b = sklearn.metrics.euclidean_distances(X, Y)
    assert_eq(a, b)

    x_norm_squared = (X ** 2).sum(axis=1).compute()[:, np.newaxis]
    a = dask_ml.metrics.euclidean_distances(X, Y, X_norm_squared=x_norm_squared)
    b = sklearn.metrics.euclidean_distances(X, Y, X_norm_squared=x_norm_squared)
    assert_eq(a, b)

    y_norm_squared = (Y ** 2).sum(axis=1).compute()[np.newaxis, :]
    a = dask_ml.metrics.euclidean_distances(X, Y, Y_norm_squared=y_norm_squared)
    b = sklearn.metrics.euclidean_distances(X, Y, Y_norm_squared=y_norm_squared)
    assert_eq(a, b)


def test_euclidean_distances_same():
    X = da.random.uniform(size=(100, 4), chunks=50)
    a = dask_ml.metrics.euclidean_distances(X, X)
    b = sklearn.metrics.euclidean_distances(X, X)
    assert_eq(a, b, atol=1e-4)

    a = dask_ml.metrics.euclidean_distances(X)
    b = sklearn.metrics.euclidean_distances(X)
    assert_eq(a, b, atol=1e-4)

    x_norm_squared = (X ** 2).sum(axis=1).compute()[:, np.newaxis]
    assert_eq(X, X, Y_norm_squared=x_norm_squared, atol=1e-4)


@pytest.mark.parametrize("kernel", ["linear", "polynomial", "rbf", "sigmoid"])
def test_pairwise_kernels(kernel):
    X = da.random.uniform(size=(100, 4), chunks=(50, 4))
    a = dask_ml.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]
    b = sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]

    r1 = a(X)
    r2 = b(X.compute())
    assert isinstance(X, da.Array)
    assert_eq(r1, r2)


@pytest.mark.parametrize("sample_weight", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("labels", [[0, 1], [0, 1, 3], [1, 0]])
@pytest.mark.parametrize("daskify", [True, False])
def test_log_loss(labels, normalize, sample_weight, daskify):
    n = 100
    c = 25
    y_true = np.random.choice(labels, size=n)
    y_pred = np.random.uniform(size=(n, len(labels)))
    y_pred /= y_pred.sum(1, keepdims=True)

    if sample_weight:
        sample_weight = np.random.uniform(size=n)
        sample_weight /= sample_weight.sum()
        dsample_weight = da.from_array(sample_weight, chunks=c)
    else:
        sample_weight = None
        dsample_weight = None

    if daskify:
        dy_true = da.from_array(y_true, chunks=c)
        dy_pred = da.from_array(y_pred, chunks=c)
    else:
        dy_true = y_true
        dy_pred = y_pred
        (dsample_weight,) = dask.compute(dsample_weight)

    a = sklearn.metrics.log_loss(
        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
    )
    b = dask_ml.metrics.log_loss(
        dy_true,
        dy_pred,
        labels=labels,
        normalize=normalize,
        sample_weight=dsample_weight,
    )

    assert_eq(a, b)


@pytest.mark.parametrize(
    "yhat",
    [
        da.from_array(np.array([0.25, 0.25, 0.75, 0.75]), chunks=2),
        da.from_array(np.array([0, 0, 1, 1]), chunks=2),
        da.from_array(
            np.array([[0.75, 0.25], [0.75, 0.25], [0.25, 0.75], [0.25, 0.75]]), chunks=2
        ),
    ],
)
def test_log_loss_shape(yhat):
    y = da.from_array(np.array([0, 0, 1, 1]), chunks=2)
    labels = [0, 1]
    a = sklearn.metrics.log_loss(y, yhat)
    b = dask_ml.metrics.log_loss(y, yhat, labels=labels)
    assert_eq(a, b)


@pytest.mark.parametrize("y", [[0, 1, 1, 0], [0, 1, 2, 0]])
def test_log_loss_scoring(y):
    # a_scorer = sklearn.metrics.get_scorer('neg_log_loss')
    # b_scorer = dask_ml.metrics.get_scorer('neg_log_loss')
    X = da.random.uniform(size=(4, 2), chunks=2)
    labels = np.unique(y)
    y = da.from_array(np.array(y), chunks=2)

    a_scorer = sklearn.metrics.make_scorer(
        sklearn.metrics.log_loss,
        greater_is_better=False,
        needs_proba=True,
        labels=labels,
    )
    b_scorer = sklearn.metrics.make_scorer(
        dask_ml.metrics.log_loss,
        greater_is_better=False,
        needs_proba=True,
        labels=labels,
    )

    clf = dask_ml.wrappers.ParallelPostFit(
        sklearn.linear_model.LogisticRegression(
            n_jobs=1, solver="lbfgs", multi_class="auto"
        )
    )
    clf.fit(*dask.compute(X, y))

    result = b_scorer(clf, X, y)
    expected = a_scorer(clf, *dask.compute(X, y))

    assert_eq(result, expected)
