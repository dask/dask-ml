.. _iid-estimators:

Meta-estimators for IID Data
===========================

dask-ml provides some meta-estimators that are appropriate for independent and
identically distributed (IID) data. These meta-estimators simplify workflows
involving larger than memory datasets. They are born out of the observations
that

1. Oftentimes, a random sample of the dataset is representative of the whole
2. Your `learning curve`_ may have leveled off before you run out of RAM
3. It's common to *train* on a small subset of the data and *predict* for a
   much larger (possibly larger than memory) dataset.

They work by training on subsets of the data, allowing you to use any
scikit-learn estimator you would use normally, and transform and predict on the
entire dataset.

Background on IID Data
----------------------

.. warning::

   Using these meta-estimators on non-IID data will bias your results.

Informally, data are IID if the order "doesn't matter". That is, if any random
sample of the data is just as likely to be representative of the whole as any
other random sample. Many datasets are *not* IID, and so the estimators in this
sub-packge *will give you biased results*. In particular, time series data are
often not IID, since there are often correlations between subsequent
observations in time series (for example, there may be a roughly constant growth
factor). If your dataset is sorted in any way (say by user ID, or by output
class) then you should shuffle your data before using these meta-estimators.

.. currentmodule:: dask_ml

.. autosummary::

   iid.FirstBlockFitter

Training on a Subset of the Data
--------------------------------

:class:`iid.FirstBlockFitter` is a meta-estimator that trains on a subset of the
data passed to ``fit``. In particular, it trains on just the first block of a
dask array, or just the first partition of a dask dataframe. Let's make a
dataset with 10,000 samples, broken into 10 blocks of 1,000 samples each.

.. ipython:: python

   import sklearn.ensemble
   from dask_ml.datasets import make_classification
   from dask_ml.iid import FirstBlockFitter

   X, y = make_classification(n_samples=10000, chunks=1000)
   X

We'll create our estimator, which wraps
:ref:`sklearn.ensemble.RandomForestClassifier`.

.. ipython:: python

   clf = FirstBlockFitter(sklearn.ensemble.RandomForestClassifier())

Training follows the usual API but is only done on the first block, 1,000
samples in this case.

.. ipython:: python

   clf.fit(X, y)

All the learned attributes are now available on ``clf``,

.. ipython:: python

   clf.classes_

When passed a dask object as input, ``transform``, ``predict``, and
``predict_proba`` return a dask object as output.

.. ipython:: python

   clf.predict_proba(X)

This can later be computed in parallel (potentially distributed on a cluster).

.. ipython:: python

   clf.predict_proba(X).compute()

Comparison to Incremental Learners
''''''''''''''''''''''''''''''''''

``dask_ml`` also offers some :ref:`incremental-learning` estimators. There's
some overlap between these two styles of estimators. The primary difference lies
in *which data the estimator is trained on*.

* Incremental estimators are trained on the *entire* dataset, one block or
  partition at a time
* IID meta-estimators are trained on *a subset of the data*

In general, an incremental estimator will give you a more accurate estimator
(especially if your data are not IID, in which case you shouldn't be using the
IID meta-estimators anyway) but will take longer to train.

Only some algorithms can be implemented to work incrementally; there may not be an
incremental version of the estimator you want to use. The IID meta-estimators
can wrap any scikit-learn estimator.

Comparison to other Estimators in dask-ml
'''''''''''''''''''''''''''''''''''''''''

``dask-ml`` re-implements some estimators from scikit-learn, for example
:ref:`cluster.KMeans`, or :ref:`preprocessing.QuantileTransformer`. For these
cases, we'd generally recommend using the re-implemented version. These have
been written specifically for dask collections, and so may be better
parallelized or tuned for distributed computation.

.. _learning curve: http://scikit-learn.org/stable/modules/learning_curve.html
