History
=======

Dask in machine learning originally consisted of a number of smaller libraries
focused around particular sub-domains of machine learning.

-  dask-searchcv_: Scalable model selection
-  dask-glm_: Generalized Linear Model solvers
-  dask-xgboost_: Connection to the XGBoost library
-  dask-tensorflow_: Connection to the Tensorflow library

While these special-purpose libraries were convenient for development, they
were inconvenient for users who found the number of libraries daunting.  The
``dask-ml`` project started as a combination of these that presented a single
unified API and entry-point that mimicked Scikit-Learn.  Afterwards additional
algorithm development happened in the ``dask-ml`` library itself.

The pre-existing libraries are still valid and dask-ml defers to them for
future development.

.. _dask-searchcv: https://github.com/dask/dask-searchcv
.. _dask-glm: https://github.com/dask/dask-glm
.. _dask-xgboost: https://github.com/dask/dask-xgboost
.. _dask-tensorflow: https://github.com/dask/dask-tensorflow
