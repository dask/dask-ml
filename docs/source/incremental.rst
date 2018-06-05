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

   estimator = SGDClassifier(random_state=10)
   clf = Incremental(estimator, classes=[0, 1])
   clf.fit(X, y)

In this example, we make a (small) random Dask Array. It has 100 samples,
broken in the 4 blocks of 25 samples each. The chunking is only along the
first axis (the samples). There is no chunking along the features.

You instantite the underlying estimator as usual. It really is just a
scikit-learn compatible estimator, and will be trained normally via its
``partial_fit``.

When wrapping the estimator in :class:`Incremental`, you need to pass any
keyword arguments that are expected by the underlying ``partial_fit`` method.
With :class:`sklearn.linear_model.SGDClassifier`, we're required to provide
the list of unique ``classes`` in ``y``.

Notice that we call the regular ``.fit`` method for training. Dask-ML takes
care of passing each block to the underlying estimator for you.

.. _incremental learning: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
