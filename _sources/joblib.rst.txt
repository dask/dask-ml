.. _joblib:

Scikit-Learn & Joblib
=====================

Many Scikit-Learn algorithms are written for parallel execution using
`Joblib <http://joblib.readthedocs.io/en/latest/>`__, which natively provides
thread-based and process-based parallelism.  Joblib is what backs the
``n_jobs=`` parameter in normal use of Scikit-Learn.

Dask can scale these Joblib-backed algorithms out to a cluster of machines by
providing an alternative Joblib backend.  The following video demonstrates how
to use Dask to parallelize a grid search across a cluster.

.. raw:: html

    <iframe width="560"
            height="315"
            src="https://www.youtube.com/embed/5Zf6DQaf7jk"
            frameborder="0"
            allow="autoplay; encrypted-media"
            allowfullscreen>
    </iframe>

To use the Dask backend to Joblib you have to create a Client, and wrap your
code with ``joblib.parallel_backend('dask')``.

.. code-block:: python

   from dask.distributed import Client
   import joblib

   client = Client(processes=False)             # create local cluster
   # client = Client("scheduler-address:8786")  # or connect to remote cluster

   with joblib.parallel_backend('dask'):
       # Your scikit-learn code

As an example you might distribute a randomized cross validated parameter
search as follows:

.. code-block:: python

   import numpy as np
   from dask.distributed import Client

   import joblib
   from sklearn.datasets import load_digits
   from sklearn.model_selection import RandomizedSearchCV
   from sklearn.svm import SVC

   client = Client(processes=False)             # create local cluster

   digits = load_digits()

   param_space = {
       'C': np.logspace(-6, 6, 13),
       'gamma': np.logspace(-8, 8, 17),
       'tol': np.logspace(-4, -1, 4),
       'class_weight': [None, 'balanced'],
   }

   model = SVC(kernel='rbf')
   search = RandomizedSearchCV(model, param_space, cv=3, n_iter=50, verbose=10)

   with joblib.parallel_backend('dask'):
       search.fit(digits.data, digits.target)


Note that the Dask joblib backend is useful for scaling out CPU-bound workloads;
workloads with datasets that fit in RAM, but have many individual operations
that can be done in parallel. To scale out to RAM-bound workloads
(larger-than-memory datasets) use one of the following alternatives:

* :ref:`parallel-meta-estimators`
* :ref:`hyperparameter.incremental`
* or one of the estimators from the :ref:`api`
