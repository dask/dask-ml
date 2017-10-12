Joblib
======

Scikit-Learn parallelizes internally using `joblib`_. ``dask.distributed``
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

.. _joblib: https://pythonhosted.org/joblib/
