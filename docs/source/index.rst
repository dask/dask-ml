.. dask-ml documentation master file, created by
   sphinx-quickstart on Fri Jun  9 09:41:40 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
dask-ml
=======

This is a repository to collect all the ways dask can be used in parallel and
distributed machine learning workflows.

Single Machine
--------------

If your computation fits on a single machine, you may be able to use dask to
speed up the computation. See here.

Distributed Learning
--------------------

Dask can help in cases where you'd like to fit a model on a dataset that doesn't
fit in memory. You can either use a library like `dask-glm`_, that's built on top
of dask, or use another distributed machine learning framework like `XGBoost`_ or `tensorflow`_.
See here.

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
