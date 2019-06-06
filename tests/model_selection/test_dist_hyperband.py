
import dask.array as da
import numpy as np
from distributed.utils_test import cluster, gen_cluster, loop  # noqa: F401
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection import HyperbandSearchCV


#  @gen_cluster(client=True, timeout=5000)
#  def test_basic1(c, s, a, b):
def _test_basic1(c, s, a, b):
    n, d = (50, 2)
    max_iter = 9

    rng = da.random.RandomState(42)

    # create observations we know linear models can fit
    X = rng.normal(size=(n, d), chunks=n // 2)
    coef_star = rng.uniform(size=d, chunks=d)
    y = da.sign(X.dot(coef_star))

    params = {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "eta0": np.logspace(-2, 0, num=1000),
    }
    model = SGDClassifier(tol=-np.inf, penalty="elasticnet", random_state=42, eta0=0.1)
    search = HyperbandSearchCV(model, params, max_iter=max_iter, random_state=42)
    classes = c.compute(da.unique(y))
    yield search.fit(X, y, classes=classes)

    assert hasattr(search, "cv_results_")


#  @gen_cluster(client=True, timeout=5000)
#  def test_basic2(c, s, a, b):
def _test_basic2(c, s, a, b):
    n, d = (50, 2)
    max_iter = 9

    rng = da.random.RandomState(42)

    # create observations we know linear models can fit
    X = rng.normal(size=(n, d), chunks=n // 2)
    coef_star = rng.uniform(size=d, chunks=d)
    y = da.sign(X.dot(coef_star))

    params = {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "eta0": np.logspace(-2, 0, num=1000),
    }
    model = SGDClassifier(tol=-np.inf, penalty="elasticnet", random_state=42, eta0=0.1)
    search = HyperbandSearchCV(model, params, max_iter=max_iter, random_state=42)
    classes = c.compute(da.unique(y))
    yield search.fit(X, y, classes=classes)

    assert hasattr(search, "cv_results_")


@gen_cluster(client=True, timeout=5000)
def test(c, s, a, b):
    _test_basic1(c, s, a, b)
    _test_basic2(c, s, a, b)
