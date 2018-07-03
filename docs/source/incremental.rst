.. _incremental-learning:

Incremental Learning
====================

Some estimators can be trained incrementally -- without seeing the entire
dataset at once. Scikit-Learn provdes the ``partial_fit`` API to stream batches
of data to an estimator that can be fit in batches.

Normally, if you pass a Dask Array to an estimator expecting a NumPy array,
the Dask Array will be converted to a single, large NumPy array. On a single
machine, you'll likely run out of RAM and crash the program. On a distributed
cluster, all the workers will send their data to a single machine and crash it.

:class:`dask_ml.wrappers.Incremental` provides a bridge between Dask and
Scikit-Learn estimators supporting the ``partial_fit`` API. You wrap the
underlying estimator in ``Incremental``. Dask-ML will sequentially pass each
block of a Dask Array to the underlying estimator's ``partial_fit`` method.

.. _incremental.blockwise-metaestimator:

Incremental Meta-estimator
--------------------------

.. currentmodule::  dask_ml

.. autosummary::
   wrappers.Incremental

:class:`dask_ml.wrappers.Incremental` is a meta-estimator (an estimator that
takes another estimator) that bridges scikit-learn estimators expecting
NumPy arrays, and users with large Dask Arrays.

Each *block* of a Dask Array is fed to the underlying estiamtor's
``partial_fit`` method. The training is entirely sequential, so you won't
notice massive training time speedups from parallelism. In a distributed
environment, you should notice some speedup from avoiding extra IO, and the
fact that models are typically much smaller than data, and so faster to move
between machines.


.. ipython:: python

   from dask_ml.datasets import make_classification
   from dask_ml.wrappers import Incremental
   from sklearn.linear_model import SGDClassifier

   X, y = make_classification(chunks=25)
   X

   estimator = SGDClassifier(random_state=10, max_iter=1000)
   clf = Incremental(estimator, scoring='accuracy')
   clf.fit(X, y, classes=[0, 1])

In this example, we make a (small) random Dask Array. It has 100 samples,
broken in the 4 blocks of 25 samples each. The chunking is only along the
first axis (the samples). There is no chunking along the features.

You instantiate the underlying estimator as usual. It really is just a
scikit-learn compatible estimator, and will be trained normally via its
``partial_fit``.

Notice that we call the regular ``.fit`` method, not ``partial_fit`` for
training. Dask-ML takes care of passing each block to the underlying estimator
for you.

Just like :meth:`sklearn.linear_model.SGDClassifier.partial_fit`, we need to
pass the ``classes`` argument to ``fit``. In general, any argument that is
required for the underlying estimators ``parital_fit`` becomes required for
the wrapped ``fit``.


.. note::

   Note that we specified ``scoring='accuracy'`` when creating the
   :class:`dask_ml.wrappers.Incremental`. Specifing `scoring` is vital
   when working with large datasets.

   The default behavior, if `scoring` is *not* specified, is to use the
   ``score`` method of the underlying estimator. Most Scikit-Learn estimators
   will accept Dask arrays, but convert them to a single, large NumPy array.
   For large datasets, this will likely exhaust the memory of your worker.


We can get the accuracy score on our dataset.

.. ipython:: python

   clf.score(X, y)

Because we specified ``scoring='accuracy'``, this uses
:meth:`dask_ml.metrics.accuracy_score`, which safely computes the accuracy
score using the Dask array (on your cluster or out-of-core), rather than
converting to a NumPy array.

All of the attributes learned durning training, like ``coef_``, are available
on the ``Incremental`` instance.

.. ipython:: python

   clf.coef_

If necessary, the actual estimator trained is available as ``Incremental.estimator_``

.. ipython:: python

   clf.estimator_

.. _incremental learning: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning

Incremental Learning and Hyper-parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`Incremental` is a meta-estimator.
To search over the hyper-parameters of the underlying estimator, use the usual scikit-learn convention of
prefixing the parameter name with ``<name>__``. For ``Incremental``, ``name`` is always ``estimator``.


.. ipython:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {'estimator__alpha': [0.10, 10.0]}
   gs = GridSearchCV(clf, param_grid, n_jobs=-1)
   gs.fit(X, y, classes=[0, 1])


This can be mixed with :ref:`joblib` to use a cluster for training in parallel, even if you're RAM-bound.
