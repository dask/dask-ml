Tensorflow
==========

.. currentmodule:: dask_ml.tensorflow

.. autosummary::

   start_tensorflow

Tensorflow_ is a library for numerical computation that's commonly used in deep
learning. It can be run in a `distributed mode`_, and :func:`start_tensorflow`
aids in setting up the Tensorflow cluster along side your existing dask cluster.

Example
-------

Given a Dask cluster

.. code-block:: python

   from dask.distributed import Client
   client = Client('scheduler-address:8786')

Get a TensorFlow cluster, specifying groups by name

.. code-block:: python

   from dask_tensorflow import start_tensorflow
   tf_spec, dask_spec = start_tensorflow(client, ps=2, worker=4)

   >>> tf_spec
   {'worker': ['192.168.1.100:2222', '192.168.1.101:2222',
               '192.168.1.102:2222', '192.168.1.103:2222'],
    'ps': ['192.168.1.104:2222', '192.168.1.105:2222']}

This creates a ``tensorflow.train.Server`` on each Dask worker and sets up a
Queue for data transfer on each worker.  These are accessible directly as
``tensorflow_server`` and ``tensorflow_queue`` attributes on the workers.

More Complex Workflow
---------------------

Typically then we set up long running Dask tasks that get these servers and
participate in general TensorFlow compuations.

.. code-block:: python

   from dask.distributed import worker_client

   def ps_function(self):
       with worker_client() as c:
           tf_server = c.worker.tensorflow_server
           tf_server.join()

   ps_tasks = [client.submit(ps_function, workers=worker, pure=False)
               for worker in dask_spec['ps']]

   def worker_function(self):
       with worker_client() as c:
           tf_server = c.worker.tensorflow_server

           # ... use tensorflow as desired ...

   worker_tasks = [client.submit(worker_function, workers=worker, pure=False)
                   for worker in dask_spec['worker']]

One simple and flexible approach is to have these functions block on queues and
feed them data from dask arrays, dataframes, etc.


.. code-block:: python

   def worker_function(self):
       with worker_client() as c:
           tf_server = c.worker.tensorflow_server
           queue = c.worker.tensorflow_queue

           while not stopping_condition():
               batch = queue.get()
               # train with batch

And then dump blocks of numpy and pandas dataframes to these queues

.. code-block:: python

   from distributed.worker_client import get_worker
   def dump_batch(batch):
       worker = get_worker()
       worker.tensorflow_queue.put(batch)


   import dask.dataframe as dd
   df = dd.read_csv('hdfs:///path/to/*.csv')
   # clean up dataframe as necessary
   partitions = df.to_delayed()  # delayed pandas dataframes
   client.map(dump_batch, partitions)

.. _Tensorflow: https://www.tensorflow.org/
.. _distributed mode: https://www.tensorflow.org/deploy/distributed
