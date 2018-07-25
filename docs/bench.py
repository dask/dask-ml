from __future__ import division, print_function

from time import time

import distributed.joblib
import numpy as np
import sklearn.ensemble.forest
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.metrics import zero_one_loss
from sklearn.utils import check_array


def Parallel2(*args, **kwargs):
    kwargs["backend"] = None
    return Parallel(*args, **kwargs)


RANDOM_STATE = 13
SCHEDULER_ADDRESS = None


def load_data():
    # Load dataset
    print("Loading dataset...")
    data = fetch_covtype(
        download_if_missing=True, shuffle=True, random_state=RANDOM_STATE
    )
    X = check_array(data["data"], dtype=np.float32, order="C")
    y = (data["target"] != 1).astype(np.int)

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    # Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test


ESTIMATORS = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100),
}

X_train, X_test, y_train, y_test = load_data()

BACKENDS = [
    ("threading", Parallel, {}),
    (
        "dask.distributed",
        Parallel2,
        {"scheduler_host": SCHEDULER_ADDRESS, "scatter": [X_train]},
    ),
]

if __name__ == "__main__":
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print(
        "%s %d (pos=%d, neg=%d, size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            np.sum(y_train == 1),
            np.sum(y_train == 0),
            int(X_train.nbytes / 1e6),
        )
    )
    print(
        "%s %d (pos=%d, neg=%d, size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            np.sum(y_test == 1),
            np.sum(y_test == 0),
            int(X_test.nbytes / 1e6),
        )
    )

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for est_name, estimator in sorted(ESTIMATORS.items()):
        for backend, parallel, backend_kwargs in BACKENDS:
            # Monkeypatch
            sklearn.ensemble.forest.Parallel = parallel
            print("Training %s with %s backend... " % (est_name, backend), end="")
            estimator_params = estimator.get_params()

            estimator.set_params(
                **{
                    p: RANDOM_STATE
                    for p in estimator_params
                    if p.endswith("random_state")
                }
            )

            if "n_jobs" in estimator_params:
                estimator.set_params(n_jobs=-1)

            # Key for the results
            name = "%s, %s" % (est_name, backend)

            with parallel_backend(backend, **backend_kwargs):
                time_start = time()
                estimator.fit(X_train, y_train)
                train_time[name] = time() - time_start

            time_start = time()
            y_pred = estimator.predict(X_test)
            test_time[name] = time() - time_start

            error[name] = zero_one_loss(y_test, y_pred)

            print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("%s %s %s %s" % ("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 44)
    for name in sorted(error, key=error.get):
        print(
            "%s %s %s %s"
            % (
                name,
                ("%.4fs" % train_time[name]),
                ("%.4fs" % test_time[name]),
                ("%.4f" % error[name]),
            )
        )

    print()
