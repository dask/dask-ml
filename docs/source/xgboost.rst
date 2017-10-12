XGBoost
-------

Dask arrays and dataframes can be used with other distributed machine learning
libraries like `xgboost`_ and `tensorflow`_, that don't necessarily use dask
objects or schedulers internally. Still, you may prefer to use dask for
pre-processing tasks or to avoid setting up a second cluster.

- :ref:`xgboost <examples/xgboost.ipynb>`
- :ref:`tensorflow <examples/tensorflow.ipynb>`

.. _XGBoost: https://xgboost.readthedocs.io/
.. _tensorflow: https://www.tensorflow.org/
