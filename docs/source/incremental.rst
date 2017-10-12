Incremental Learning
====================

.. currentmodule::  daskml

.. autosummary::
   naive_bayes.PartialBernoulliNB
   naive_bayes.PartialMultinomialNB
   linear_model.PartialSGDRegressor
   linear_model.PartialSGDClassifier
   perceptron.PartialPerceptron
   passive_aggressive.PartialPassiveAggressiveClassifier
   passive_aggressive.PartialPassiveAggressiveRegressor
   cluster.PartialMiniBatchKMeans

Some scikit-learn models support `incremental learning`_, they can see batches
of the datasets and update the parameters as new data comes in. This fits nicely
with dask's block-wise nature: dask arrays are composed of many smaller NumPy
arrays. Dask-ML wraps scikit-learn's incremental learners, so that the usual
``.fit`` API will work on larger-than-memory datasets. These wrappers can be
dropped into a :class:`sklearn.pipeline.Pipeline` just like normal. In
Dask-ml, all of these estimators are prefixed with ``Partial``, e.g.
:class:`PartialSGDClassifier`.

.. note::

   While these wrappers are useful for fitting on larger than memory datasets
   out-of-core, they *do not* support any kind of parallelism or distributed
   learning. Inside, e.g. ``PartialSGDClassifier.fit()``, execution is entirely
   sequential.

.. _incremental learning: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
