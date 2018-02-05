"""
.. _plot_parallel_prediction_example.py:

Parallel Prediction Example
===========================

This example demonstrates :class:`wrappers.ParallelPostFit`. A
:class:`sklearn.svm.SVC` is fit on a small dataset that easily fits in memory.

After training, we predict for successively larger datasets. We compare

1. The serial prediction time using the regular ``SVC.predict`` method
2. The parallel prediction time using :meth:`ParallelPostFit.predict``

We see that the parallel version is faster, especially for larger datasets.
Additionally, the parallel version from ``ParallelPostFit`` scales out to
larger than memory datasets.

While only ``predict`` is demonstrated here, :class:`wrappers.ParallelPostFit`
is equally useful for ``predict_proba`` and ``transform``.
"""
from timeit import default_timer as tic

import pandas as pd
import seaborn as sns
import sklearn.datasets
from sklearn.svm import SVC

import dask_ml.datasets
from dask_ml.wrappers import ParallelPostFit

X, y = sklearn.datasets.make_classification(n_samples=1000)
clf = ParallelPostFit(SVC())
clf.fit(X, y)


Ns = [100_000, 200_000, 400_000, 800_000]
timings = []


for n in Ns:
    X, y = dask_ml.datasets.make_classification(n_samples=n,
                                                random_state=n,
                                                chunks=n // 20)
    t1 = tic()
    # Serial scikit-learn version
    clf.estimator.predict(X)
    timings.append(('Scikit-Learn', n, tic() - t1))

    t1 = tic()
    # Parallelized scikit-learn version
    clf.predict(X).compute()
    timings.append(('dask-ml', n, tic() - t1))


df = pd.DataFrame(timings,
                  columns=['method', 'Number of Samples', 'Predict Time'])
ax = sns.factorplot(x='Number of Samples', y='Predict Time', hue='method',
                    data=df, aspect=1.5)
