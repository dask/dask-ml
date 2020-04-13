Cross Validation
================

See the `scikit-learn cross validation documentation`_ for a fuller discussion of cross validation.
This document only describes the extensions made to support Dask arrays.

The simplest way to split one or more Dask arrays is with :func:`dask_ml.model_selection.train_test_split`:

.. ipython:: python

   import dask.array as da
   from dask_ml.datasets import make_regression
   from dask_ml.model_selection import train_test_split

   X, y = make_regression(n_samples=125, n_features=4, random_state=0, chunks=50)
   X

The interface for splitting Dask arrays is the same as scikit-learn's version.

.. ipython:: python

   X_train, X_test, y_train, y_test = train_test_split(X, y)
   X_train  # A dask Array

   X_train.compute()[:3]


While it's possible to pass dask arrays to :func:`sklearn.model_selection.train_test_split`, we recommend
using the Dask version for performance reasons: the Dask version is faster
for two reasons:

First, **the Dask version shuffles blockwise**.
In a distributed setting, shuffling *between* blocks may require sending large amounts of data between machines, which can be slow.
However, if there's a strong pattern in your data, you'll want to perform a full shuffle.

Second, the Dask version avoids allocating large intermediate NumPy arrays storing the indexes for slicing.
For very large datasets, creating and transmitting ``np.arange(n_samples)`` can be expensive.

.. _scikit-learn cross validation documentation: http:/scikit-learn.org/stable/modules/cross_validation.html
