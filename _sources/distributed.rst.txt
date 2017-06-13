.. _distributed:

====================
Distributed Learning
====================

Dask is able to simplify distributed machine learning.

dask-glm
--------

`dask-glm`_ is a library for fitting Generalized Linear Models that's built on
top of dask. Typically, you'd use it like a regular scikit-learn estimator:
instantiate an estimator and call ``estimator.fit(X, y)`` with dask arrays.
See :ref:`examples/dask-glm.ipynb` for an example.

distributed-joblib
------------------

``scikit-learn`` parallelizes internally using `joblib`_. ``dask.distributed``
registers itself with ``joblib``, allowing you to use your cluster to fit a
scikit-learn estimator. Given a scikit-learn estimator called ``estimator`` and
some data ``X`` and ``y``, you'd fit it on a single machine (in parallel) like

.. code-block:: python

   >>> estimator.fit(X, y)


Assuming you have a ``distributed.Client`` called ``client``, this can be fit on
your cluster like

.. code-block:: python

   >>> from sklearn.externals import joblib
   >>> import distributed.joblib
   >>> with joblib.parallel_backend('dask.distributed',
   ...                               scheduler_host=client.scheduler.address):
   ...     estimator.fit(X, y)

See :ref:`here <examples/joblib-distributed.ipynb>` for an example.

Interoperate
------------

dask arrays can be used with other distributed machine learning libraries like
`xgboost`_ and `tensorflow`_, that don't necessarily use dask objects or
schedulers internally. Still, you may prefer to use dask for pre-processing
tasks or to avoid setting up a second cluster.

- :ref:`xgboost <examples/xgboost.ipynb>`
- :ref:`tensorflow <examples/tensorflow.ipynb>`

.. _dask-glm: http://dask-glm.readthedocs.io/
.. _XGBoost: https://xgboost.readthedocs.io/
.. _tensorflow: https://www.tensorflow.org/
.. _joblib: https://pythonhosted.org/joblib/
