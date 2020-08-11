.. _incremental-learning:

Incremental Learning
====================

Some estimators can be trained incrementally -- without seeing the entire
dataset at once. Scikit-Learn provides the ``partial_fit`` API to stream batches
of data to an estimator that can be fit in batches.

Normally, if you pass a Dask Array to an estimator expecting a NumPy array,
the Dask Array will be converted to a single, large NumPy array. On a single
machine, you'll likely run out of RAM and crash the program. On a distributed
cluster, all the workers will send their data to a single machine and crash it.

:class:`dask_ml.wrappers.Incremental` provides a bridge between Dask and
Scikit-Learn estimators supporting the ``partial_fit`` API. You wrap the
underlying estimator in ``Incremental``. Dask-ML will sequentially pass each
block of a Dask Array to the underlying estimator's ``partial_fit`` method.

.. note::

   :class:`dask_ml.wrappers.Incremental` currently does not work well with
   hyper-parameter optimization like :class:`sklearn.model_selection.GridSearchCV`.
   If you need to do hyper-parameter optimization on larger-than-memory datasets,
   we recommend :class:`dask_ml.model_selection.IncrementalSearchCV`. See
   ":ref:`hyperparameter.incremental`" for an introduction.

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

   estimator = SGDClassifier(random_state=10, max_iter=100)
   clf = Incremental(estimator)
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

   Take care with the behavior of :meth:`Incremental.score`. Most estimators
   inherit the default scoring methods of R2 score for regressors and accuracy
   score for classifiers. For these estimators, we automatically use Dask-ML's
   scoring methods, which are able to operate on Dask arrays.

   If your underlying estimator uses a different scoring method, you'll need
   to ensure that the scoring method is able to operate on Dask arrays. You
   can also explicitly pass ``scoring=`` to pass a dask-aware scorer.

We can get the accuracy score on our dataset.

.. ipython:: python

   clf.score(X, y)

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

See ":ref:`hyperparameter.incremental`" for more on how to do hyperparameter optimization on
larger than memory datasets.
