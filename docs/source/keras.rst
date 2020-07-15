Keras
=====

The package SciKeras_ brings a Scikit-learn API to Keras. Install directions
are at https://github.com/adriangb/scikeras/blob/master/README.md#installation.

Example usage
-------------

First, let's start by defining normal function to create our model. This is the
normal way to create a `Keras Sequential model`_

.. _Keras Sequential model: https://keras.io/api/models/sequential/

.. code-block:: python

   import tensorflow as tf
   from tensorflow.keras.layers import Dense, Activation, Dropout
   from tensorflow.keras.models import Sequential

   def _keras_build_fn(lr=0.01):
       layers = [Dense(512, input_shape=(784,), activation="relu"),
                 Dense(10, input_shape=(512,), activation="softmax")]
       model = Sequential(layers)

       opt = tf.keras.optimizers.SGD(learning_rate=lr)
       model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
       return model

Now, we can use the SciKeras to create a Scikit-learn compatible model:

.. code-block:: python

   from scikeras.wrappers import KerasClassifier, KerasRegressor
   model = KerasClassifier(build_fn=_keras_build_fn, lr=0.1)

This model will work with all of Dask-ML: it expects NumPy arrays as inputs and
obeys the Scikit-learn API. For example, the following examples have been
implemented:

* Using Keras with the relevant model selection in Dask-ML, :class:`~dask_ml.model_selection.HyperbandSearchCV`
  (TODO: link).
* Using Keras with Dask-ML's :class:`~dask_ml.wrappers.Incremental` (TODO link)

.. _SciKeras: https://github.com/adriangb/scikeras
