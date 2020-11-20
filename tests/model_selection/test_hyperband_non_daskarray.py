import numpy as np
import pandas as pd

from dask_ml.model_selection import HyperbandSearchCV
from dask_ml.datasets import make_classification
from distributed.utils_test import gen_cluster
from sklearn.linear_model import SGDClassifier


@gen_cluster(client=True)
def test_pandas():
    X, y = make_classification(chunks=100)
    X, y = pd.DataFrame(X.compute()), pd.Series(y.compute())

    est = SGDClassifier(tol=1e-3)
    param_dist = {'alpha': np.logspace(-4, 0, num=1000),
                  'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                  'average': [True, False]}

    search = HyperbandSearchCV(est, param_dist)
    search.fit(X, y, classes=y.unique())
    assert search.best_params_
