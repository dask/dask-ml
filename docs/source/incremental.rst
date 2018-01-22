Incremental Learning
====================

.. currentmodule::  dask_ml

.. autosummary::
   naive_bayes.PartialBernoulliNB
   naive_bayes.PartialMultinomialNB
   linear_model.PartialSGDRegressor
   linear_model.PartialSGDClassifier
   linear_model.PartialPerceptron
   linear_model.PartialPassiveAggressiveClassifier
   linear_model.PartialPassiveAggressiveRegressor
   cluster.PartialMiniBatchKMeans
   base._BigPartialFitMixin

Scikit-Learn's Partial Fit
--------------------------

Some scikit-learn models support `incremental learning`_ with the
``.partial_fit`` API.  These models can see small batches of dataset and update
their parameters as new data arrives.

.. code-block:: python

    for X_block, y_block in iterator_of_numpy_arrays:
        est.partial_fit(X_block, y_block)

This block-wise learning fits nicely with Dask's block-wise nature: Dask
arrays are composed of many smaller NumPy arrays.  Dask dataframes and arrays
provides an intuitive way to preprocess your data and then intuitively send
that data to an incremental model piece by piece.  Dask-ML will hide the
``.partial_fit`` mechanics from you, so that the usual ``.fit`` API will work
on larger-than-memory datasets. These wrappers can be dropped into a
:class:`sklearn.pipeline.Pipeline` just like normal. In Dask-ml, all of these
estimators are prefixed with ``Partial``, e.g.  :class:`PartialSGDClassifier`.

.. note::

   While these wrappers are useful for fitting on larger than memory datasets
   they do not offer any kind of parallelism while training.  Calls to
   ``.fit()`` will be entirely sequential.

Example
-------

.. ipython:: python

   from dask_ml.linear_model import PartialSGDRegressor
   from dask_ml.datasets import make_classification
   X, y = make_classification(n_samples=1000, chunks=500)
   est = PartialSGDRegressor()
   est.fit(X, y)

.. _incremental learning: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
