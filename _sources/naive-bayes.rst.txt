Naive Bayes
===========

.. currentmodule:: dask_ml.naive_bayes

.. autosummary::
   GaussianNB


See `the scikit-learn documentation <https://scikit-learn.org/stable/modules/naive_bayes.html>`_ for more information about Naive Bayes classifiers.

These follow the scikit-learn estimator API, and so can be dropped into
existing routines like grid search and pipelines, but are implemented
externally with new, scalable algorithms and so can consume distributed Dask
arrays and DataFrames rather than just in-memory NumPy and Pandas arrays
and DataFrames.

Example
-------

.. ipython:: python
   :okwarning:

   from dask_ml import datasets
   from dask_ml.naive_bayes import GaussianNB
   X, y = datasets.make_classification(chunks=50)
   gnb = GaussianNB()
   gnb.fit(X, y)
