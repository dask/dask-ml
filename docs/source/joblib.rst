.. _joblib:

Joblib
======

Dask.distributed integrates with `Joblib <http://joblib.readthedocs.io/en/latest/>`__
by providing an alternative
cluster-computing backend, alongside Joblib's builtin threading and
multiprocessing backends. This enables training a scikit-learn model in
parallel using a cluster of machines.

The following video demonstrates how to use Dask to parallelize a grid
search across a cluster.

.. raw:: html

    <iframe width="560"
            height="315"
            src="https://www.youtube.com/embed/5Zf6DQaf7jk"
            frameborder="0"
            allow="autoplay; encrypted-media"
            allowfullscreen>
    </iframe>

`Joblib <http://joblib.readthedocs.io/en/latest/>`__
is a library for simple parallel programming primarily developed and
used by the Scikit Learn community.  As of version 0.10.0 it contains a plugin
mechanism to allow Joblib code to use other parallel frameworks to execute
computations.  The ``dask.distributed`` scheduler implements such a plugin in
the ``dask_ml.joblib`` module and registers it appropriately with Joblib
when imported.  As a result, any joblib code (including many scikit-learn
algorithms) will run on the distributed scheduler if you enclose it in a
context manager as follows:

.. code-block:: python

   import dask_ml.joblib  # registers joblib plugin
   from dask.distributed import Client

   client = Client(processes=False)             # create local cluster
   # client = Client("scheduler-address:8786")  # or connect to remote cluster

   from joblib import Parallel, parallel_backend

   with parallel_backend('dask'):
       # normal Joblib code

Note that scikit-learn bundles joblib internally, so if you want to specify the
joblib backend you'll need to import ``parallel_backend`` from scikit-learn
instead of ``joblib``. As an example you might distributed a randomized cross
validated parameter search as follows.

.. code-block:: python

   import dask_ml.joblib  # registers joblib plugin
   # Scikit-learn bundles joblib, so you need to import from
   # `sklearn.externals.joblib` instead of `joblib` directly
   from sklearn.externals.joblib import parallel_backend
   from sklearn.datasets import load_digits
   from sklearn.grid_search import RandomizedSearchCV
   from sklearn.svm import SVC
   import numpy as np

   digits = load_digits()

   param_space = {
       'C': np.logspace(-6, 6, 13),
       'gamma': np.logspace(-8, 8, 17),
       'tol': np.logspace(-4, -1, 4),
       'class_weight': [None, 'balanced'],
   }

   model = SVC(kernel='rbf')
   search = RandomizedSearchCV(model, param_space, cv=3, n_iter=50, verbose=10)

   with parallel_backend('dask'):
       search.fit(digits.data, digits.target)


For large arguments that are used by multiple tasks, it may be more efficient
to pre-scatter the data to every worker, rather than serializing it once for
every task. This can be done using the ``scatter`` keyword argument, which
takes an iterable of objects to send to each worker.

.. code-block:: python

   # Serialize the training data only once to each worker
   with parallel_backend('dask', scatter=[digits.data, digits.target]):
       search.fit(digits.data, digits.target)
