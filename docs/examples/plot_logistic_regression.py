"""
.. _plot_logistic_regression_example.py:

Logistic Regression Example
===========================

Comparison of scaling.
"""
from dask_ml.datasets import make_classification
import pandas as pd

from timeit import default_timer as tic
import sklearn.linear_model
import dask_ml.linear_model
import seaborn as sns

Ns = [2500, 5000, 7500, 10000]

timings = []

for n in Ns:
    X, y = make_classification(n_samples=n, random_state=n, chunks=n // 20)
    t1 = tic()
    sklearn.linear_model.LogisticRegression().fit(X, y)
    timings.append(('Scikit-Learn', n, tic() - t1))
    t1 = tic()
    dask_ml.linear_model.LogisticRegression().fit(X, y)
    timings.append(('dask-ml', n, tic() - t1))


df = pd.DataFrame(timings, columns=['method', 'Number of Samples', 'Fit Time'])
sns.factorplot(x='Number of Samples', y='Fit Time', hue='method',
               data=df, aspect=1.5)
