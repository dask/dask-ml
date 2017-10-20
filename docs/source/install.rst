Installation
============

Conda
~~~~~

``dask-ml`` is available on conda-forge and can be installed with

.. code-block:: none

   conda install -c conda-forge dask-ml

PyPI
~~~~

Wheels and a source distribution are available on conda-forge and can be
installed with

.. code-block:: none

   pip install dask-ml


Optional Dependencies
---------------------

``dask_ml.xgboost`` requires ``xgboost`` and ``dask-xgboost``. Both of these
are available on conda-forge and PyPI.

``dask_ml.tensorflow`` requires ``tensorflow`` and ``dask-tensorflow``. Both of
these are available on conda-forge and PyPI.

The conda-forge package will install all optional dependencies. With pip, the
optional dependencies can be installed as

.. code-block:: none

   pip install dask-ml[xgboost]  # also install xgboost and dask-xgboost
   pip install dask-ml[tensorflow]
   pip install dask-ml[complete]  # install all optional dependencies
