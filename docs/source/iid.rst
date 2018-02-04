Estimators for IID Data
=======================

.. warning::

   Only the first block or partition of your dask object is used for training.

The ``dask_ml.iid`` sub-package offers *many* estimators for data that are
independently and identically distributed (IID). The estimators in this
sub-package are suitable large datasets stored as dask Arrays or DataFrames *if
your data are IID*.

This sub-package was born out of the observation that

1. Oftentimes, a random sample of the dataset is representative of the whole
2. Your *learning curve* may have leveled off before you run out of RAM (TODO:
   link)
3. It's common to *train* on a small subset of the data and *predict* for a
   much larger dataset.

The ``dask_ml.iid`` package layout is identical to scikit-learn's. For example,
we'll import ``RandomForestClassier`` from ``dask_ml.iid.ensemble``.

.. ipython:: python

   from dask_ml.iid.ensemble import RandomForestClassifier
   from dask_ml.datasets import make_classification

   X, y = make_classification(n_samples=10000, chunks=1000,
                              random_state=0)

   clf = RandomForestClassifier()
   clf.fit(X, y)

This has trained on just the first block (1000 samples in this case). But we can
easily predict for the entire dataset, returning a dask array.

.. ipython:: python

   clf.predict(X)
   
IID Data
''''''''

Informally, data are IID if the order "doesn't matter". That is, if any random
sample of the data is just as likely to be representative of the whole as any
other random sample. Many datasets are *not* IID, and so the estimators in this
sub-packge *will give you biased results*. In particular, time series data are
often not IID, since there are often correlations between subsequent
observations in time series (for example, there may be a roughly constant growth
factor). If your dataset is sorted in any way (say by user ID, or by output
class) then you should shuffle your data (TODO: using...)

Compared to the scikit-learn implementation, the following methods return dask
arrays or dataframes, instead of NumPy or pandas versions:

* ``.predict``
* ``.transform``

The general rule is that a dask collection is returned if the input was a dask
collection, and the output is large (same length as the input).

Comparison to Incremental Learners
''''''''''''''''''''''''''''''''''

``dask_ml`` also offers many :ref:<incremental `incremental`>_ estimators.
There's some overlap between these two styles of estimators. The primary
difference lies in *which data the estimator is trained on*.

* Incremental estimators are trained on the *entire* dataset, one block or
  partition at a time
* IID estimators are trained on *only the first block or partition*.

In general, an incremental estimator will give you a more accurate estimator
(especially if your data are not IID) but will take longer to train.
Additionally, only some algorithms can be trained incrementally; there may not
be an incremental version of the estimator you want to use.

Comparison to other Estimators in dask-ml
'''''''''''''''''''''''''''''''''''''''''

``dask-ml`` re-implements some estimators from scikit-learn, for example
``KMeans``, or ``QuantileTransformer``. For these cases, we'd generally
recommend using the re-implemented version. These have been written specifically
for dask collections, and so may be better parallelized or tuned for distributed
computation.
