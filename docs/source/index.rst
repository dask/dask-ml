.. _dask-ml

=======
dask-ml
=======

This is a repository to collect all the ways dask can be used in parallel and
distributed machine learning workflows. There are quite a few options, and the
best one depends on your goals, data, and available compute.

Single Machine
--------------

If your computation fits on a single machine, you may be able to use dask to
speed up the computation. See :ref:`single-machine`.

Distributed Learning
--------------------

Dask can help in cases where you'd like to fit a model on a dataset that doesn't
fit in memory. You can either use a library like `dask-glm`_, that's built on top
of dask, or use another distributed machine learning framework like `XGBoost`_ or `tensorflow`_.
See :ref:`distributed`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   single-machine.rst
   distributed.rst
   examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _dask-glm: http://dask-glm.readthedocs.io/
.. _XGBoost: https://xgboost.readthedocs.io/
.. _tensorflow: https://www.tensorflow.org/
