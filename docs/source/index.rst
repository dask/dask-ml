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
   :hidden:
   :caption: Get Started

   install.rst
   examples.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Use

   preprocessing.rst
   hyper-parameter-search.rst
   glm.rst
   joblib.rst
   meta-estimators.rst
   incremental.rst
   clustering.rst
   xgboost.rst
   tensorflow.rst
   modules/api.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Develop

   changelog.rst
   contributing.rst
   history.rst
