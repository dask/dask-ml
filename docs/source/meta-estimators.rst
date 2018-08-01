.. _parallel-meta-estimators:

.. currentmodule:: dask_ml

Parallel Meta-estimators
========================

dask-ml provides some meta-estimators that parallelize and scaling out
certain tasks that may not be parallelized within scikit-learn itself.
For example, :class:`~wrappers.ParallelPostFit` will parallelize the
``predict``, ``predict_proba`` and ``transform`` methods, enabling them
to work on large (possibly larger-than memory) datasets.

Parallel Prediction and Transformation
''''''''''''''''''''''''''''''''''''''

:class:`wrappers.ParallelPostFit` is a meta-estimator for parallelizing
post-fit tasks like prediction and transformation. It can wrap any
scikit-learn estimator to provide parallel ``predict``, ``predict_proba``, and
``transform`` methods.

.. warning::

   ``ParallelPostFit`` does *not* parallelize the training step. The underlying
   estimator's ``.fit`` method is called normally.

Since just the ``predict``, ``predict_proba``, and ``transform`` methods are
wrapped, :class:`wrappers.ParallelPostFit` is most useful in situations where
your training dataset is relatively small (fits in a single machine's memory),
and prediction or transformation must be done on a much larger dataset (perhaps
larger than a single machine's memory).

.. ipython:: python

   from sklearn.ensemble import GradientBoostingClassifier
   import sklearn.datasets
   import dask_ml.datasets
   from dask_ml.wrappers import ParallelPostFit

In this example, we'll make a small 1,000 sample training dataset

.. ipython:: python

   X, y = sklearn.datasets.make_classification(n_samples=1000,
                                               random_state=0)

Training is identical to just calling ``estimator.fit(X, y)``. Aside from
copying over learned attributes, that's all that ``ParallelPostFit`` does.

.. ipython:: python

   clf = ParallelPostFit(estimator=GradientBoostingClassifier())
   clf.fit(X, y)

This class is useful for predicting for or transforming large datasets.
We'll make a larger dask array ``X_big`` with 10,000 samples per block.

.. ipython:: python

   X_big, _ = dask_ml.datasets.make_classification(n_samples=100000,
                                                   chunks=10000,
                                                   random_state=0)
   clf.predict(X_big)

This returned a ``dask.array``. Like any dask array, the actual ``compute`` will
cause the scheduler to compute tasks in parallel. If you've connected to a
``dask.distributed.Client``, the computation will be parallelized across your
cluster of machines.

.. ipython:: python

   clf.predict_proba(X_big).compute()[:10]

See `parallelizing prediction`_ for an example of how this
scales for a support vector classifier.

Comparison to other Estimators in dask-ml
'''''''''''''''''''''''''''''''''''''''''

``dask-ml`` re-implements some estimators from scikit-learn, for example
:class:`dask_ml.cluster.KMeans`, or :class:`dask_ml.preprocessing.QuantileTransformer`. This raises
the question, should I use the reimplemented dask-ml versions, or should I wrap
scikit-learn version in a meta-estimator? It varies from estimator to estimator,
and depends on your tolerance for approximate solutions and the size of your
training data. In general, if your training data is small, you should be fine
wrapping the scikit-learn version with a ``dask-ml`` meta-estimator.

.. _learning curve: http://scikit-learn.org/stable/modules/learning_curve.html
.. _parallelizing prediction: http://dask-ml-benchmarks.readthedocs.io/en/latest/auto_examples/plot_parallel_post_fit_scaling.html
