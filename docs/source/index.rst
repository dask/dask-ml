.. _dask-ml:

=======
dask-ml
=======

Dask-ML enables parallel and distributed machine learning using Dask_ alongside
existing machine learning libraries like Scikit-Learn_.

This library combines a variety of approaches, including the following:

1.  Accelerating existing algorithms within Scikit-Learn
2.  Implementing new parallel algorithms
3.  Deploying other distributed services like XGBoost or TensorFlow

In all cases we endeavor to provide a single unified interface around the
familiar NumPy, Pandas, and Scikit-Learn APIs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   changelog.rst
   install.rst
   contributing.rst
   preprocessing.rst
   hyper-parameter-search.rst
   glm.rst
   incremental.rst
   joblib.rst
   xgboost.rst
   tensorflow.rst
   clustering.rst
   examples.rst
   modules/api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Scikit-Learn: http://scikit-learn.org/stable/
.. _Dask: https://dask.pydata.org/en/latest/
