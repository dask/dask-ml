.. _dask-ml:

=======
dask-ml
=======

``dask-ml`` is a library for parallel and distributed machine learning.

Single Machine
^^^^^^^^^^^^^^

*``dask-ml`` helps parallelize machine learning workloads that fit on a single-machine.*

If you dataset fits in RAM, ``dask-ml`` can help you fit more models in less
time, by

1. Using all the cores available on your machine.
2. Using dask's intelligent data-hashing to avoid redundant computation

See :ref:`single-machine` for more.

Distributed Learning
^^^^^^^^^^^^^^^^^^^^

*``dask-ml`` implements distributed machine learning algorithms*

``dask.array`` and ``dask.dataframe`` scale out to process data on a cluster of
computers. ``dask-ml`` implements distributed algorithms that operate on dask
collections.

Additionally, ``dask-ml`` can peer with other distributed machine learning
framework like `XGBoost`_ or `tensorflow`_. See :ref:`distributed` for more.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   single-machine.rst
   distributed.rst
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
