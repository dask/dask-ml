.. _dask-ml:

=======
dask-ml
=======

Dask-ML enables parallel and distributed machine learning using Dask_ alongside
existing machine learning libraries like Scikit-Learn_.

Single Machine
^^^^^^^^^^^^^^

*Dask-ML helps parallelize machine learning workloads that fit on a single-machine.*

If your dataset fits in RAM, ``dask-ml`` can help you fit more models in less
time, by

1. Using all the cores available on your machine.
2. Using dask's intelligent data-hashing to avoid redundant computation

See :ref:`single-machine` for more.

Distributed Learning
^^^^^^^^^^^^^^^^^^^^

*Dask-ML implements distributed machine learning algorithms*

The Dask_ modules ``dask.array`` and ``dask.dataframe`` scale out data
processing to a cluster of computers. Dask-ML implements distributed algorithms
that operate on dask collections.

Additionally, Dask-ML can peer with other distributed machine learning
framework like `XGBoost`_ or `tensorflow`_. See :ref:`distributed` for more
information.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   single-machine.rst
   distributed.rst
   clustering.rst
   examples.rst
   modules/api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _dask-glm: http://dask-glm.readthedocs.io/
.. _XGBoost: https://xgboost.readthedocs.io/
.. _tensorflow: https://www.tensorflow.org/
.. _Scikit-Learn: http://scikit-learn.org/stable/
.. _Dask: https://dask.pydata.org/en/latest/
