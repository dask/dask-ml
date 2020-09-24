.. _keras:

Keras and Tensorflow
====================

The package SciKeras_ brings a Scikit-learn API to Keras. This allows Dask-ML
to be used seamlessly with Keras models.

Installation
------------

Following the `Tensorflow install directions`_ and `SciKeras install guide`_,
these packages need to be installed:

.. code-block:: bash

   $ pip install tensorflow>=2.3.0
   $ pip install scikeras>=0.1.8

These are the minimum versions that Dask-ML requires to use Tensorflow/Keras.

.. _Tensorflow install directions: https://www.tensorflow.org/install
.. _SciKeras install guide: https://github.com/adriangb/scikeras#installation

Usage
-----

First, let's start by defining normal function to create our model. This is the
normal way to create a `Keras Sequential model`_

.. _Keras Sequential model: https://keras.io/api/models/sequential/

.. code-block:: python

   import tensorflow as tf
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.models import Sequential

   def build_model(lr=0.01, momentum=0.9):
       layers = [Dense(512, input_shape=(784,), activation="relu"),
                 Dense(10, input_shape=(512,), activation="softmax")]
       model = Sequential(layers)

       opt = tf.keras.optimizers.SGD(
           learning_rate=lr, momentum=momentum, nesterov=True,
       )
       model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
       return model

Now, we can use the SciKeras to create a Scikit-learn compatible model:

.. code-block:: python

   from scikeras.wrappers import KerasClassifier
   niceties = dict(verbose=False)
   model = KerasClassifier(build_fn=build_model, lr=0.1, momentum=0.9, **niceties)

This model will work with all of Dask-ML: it can use NumPy arrays as inputs and
obeys the Scikit-learn API. For example, it's possible to use Dask-ML to do the
following:

* Use Keras with Dask-ML's model selection, including
  :class:`~dask_ml.model_selection.HyperbandSearchCV`.
* Use Keras with Dask-ML's :class:`~dask_ml.wrappers.Incremental`.

If we want to tune ``lr`` and ``momentum``, SciKeras requires that we pass
``lr`` and ``momentum`` at initialization:

.. code-block::

   model = KerasClassifier(build_fn=build_model, lr=None, momentum=None, **niceties)

.. _SciKeras: https://github.com/adriangb/scikeras

SciKeras supports more model creation methods, including some that are
backwards-compatible with Tensorflow. Refer to their documentation for details.

Example: Hyperparameter Optimization
------------------------------------

If we wanted to, we could use the model above with
:class:`~dask_ml.model_selection.HyperbandSearchCV`. Let's tune this model on
the MNIST dataset:

.. code-block:: python

   from tensorflow.keras.datasets import mnist
   from tensorflow.keras.utils import to_categorical
   import numpy as np
   from typing import Tuple

   def get_mnist() -> Tuple[np.ndarray, np.ndarray]:
       (X_train, y_train), _ = mnist.load_data()
       X_train = X_train.reshape(X_train.shape[0], 784)
       X_train = X_train.astype("float32")
       X_train /= 255
       return X_train, y_train

And let's perform the basic task of tuning our SGD implementation:

.. code-block:: python

   from scipy.stats import loguniform, uniform
   params = {"lr": loguniform(1e-3, 1e-1), "momentum": uniform(0, 1)}
   X, y = get_mnist()

Now, the search can be run:

.. code-block:: python

   from dask.distributed import Client
   client = Client()

   from dask_ml.model_selection import HyperbandSearchCV
   search = HyperbandSearchCV(model, params, max_iter=27)
   search.fit(X, y)
