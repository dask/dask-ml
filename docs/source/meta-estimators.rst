.. _parallel-meta-estimators:

Parallel Meta-estimators
========================

dask-ml provides some meta-estimators that parallelize certain tasks may not be
parallelized within scikit-learn itself. For example,
:class:`~wrappers.ParallelPostFit` will parallelize the predict and transform
methods.

Parallel Prediction and Transformation
''''''''''''''''''''''''''''''''''''''

:class:`wrappers.ParallelPostFist` is a meta-estimator for parallelizing
post-fit tasks like prediction and transformation. It can wrap any
scikit-learn estimator.

.. warning::

   ``ParallelPostFit`` does *not* parallelize the training step. The underlying
   estimator's ``.fit`` method is called normally.

Since just the ``predict``, ``predict_proba``, and ``transform`` methods are
wrapped, :class:`wrappers.ParallelPostFit` is most useful in situations where
your training dataset is relatively small (fits in a single machine's memory),
and prediction or transformation dataset is relatively large (perhaps larger
than a single machine's memory).

.. ipython:: python

   from sklearn.ensemble import GradientBoostingClassifier
   import sklearn.datasets
   import dask_ml.datasets

Make a small 1,000 sample 2 training dataset

.. ipython:: python

   X, y = sklearn.datasets.make_classification(n_samples=1000,
                                               random_state=0)

Wrap the regular classifier and fit normally.

.. ipython:: python

   clf = ParallelPostFit(GradientBoostingClassifier())
   clf.fit(X, y)

Learned attributes are available

.. ipython:: python

   clf.classes_

Transform and predict return dask outputs for dask inputs.

.. ipython:: python

   X_big, y_big = dask_ml.datasets.make_classification(n_samples=100000,
                                                       random_state=0)
   clf.predict(X)

Which can be computed in parallel, using all the resources of your
cluster if you've attached a ``Client``.

.. ipython:: python

   clf.predict_proba(X).compute()

Comparison to other Estimators in dask-ml
'''''''''''''''''''''''''''''''''''''''''

``dask-ml`` re-implements some estimators from scikit-learn, for example
:ref:`cluster.KMeans`, or :ref:`preprocessing.QuantileTransformer`. For these
cases, we'd generally recommend using the re-implemented version. These have
been written specifically for dask collections, and so may be better
parallelized or tuned for distributed computation.

.. _learning curve: http://scikit-learn.org/stable/modules/learning_curve.html
