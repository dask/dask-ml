.. _dask-ml:

=======
Dask-ML
=======

Dask-ML provides scalable machine learning in Python using Dask_ alongside
popular machine learning libraries like Scikit-Learn_.

You can try Dask-ML on a small cloud instance by clicking the following button:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/dask/dask-examples/master?filepath=machine-learning.ipynb

.. code-block:: python

   import dask.dataframe as dd
   df = dd.read_parquet('...')
   data = df[['age', 'income', 'married']]
   labels = df['outcome']

   from dask_ml.linear_model import LogisticRegression
   lr = LogisticRegression()
   lr.fit(data, labels)

What does this offer?
---------------------

See the navigation pane to the left for a list of categories of
functionality.

How does this work?
-------------------

Modern machine learning algorithms employ a wide variety of techniques.
Scaling these requires a similarly wide variety of different approaches.
Generally solutions fall into the following three categories:

Parallelize Scikit-Learn Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-Learn already provides parallel computing on a single machine with
`Joblib <http://joblib.readthedocs.io/en/latest/>`__.
Dask can now step in and take over this parallelism for many
Scikit-Learn estimators.  This works well for modest data sizes but large
computations, such as random forests, hyper-parameter optimization, and more.

.. code-block:: python

   from dask.distributed import Client
   client = Client()  # start a local Dask client

   import dask_ml.joblib
   from sklearn.externals.joblib import parallel_backend
   with parallel_backend('dask'):
       # Your normal scikit-learn code here

See :doc:`Dask-ML Joblib documentation <joblib>` for more information.

*Note that this is an active collaboration with the Scikit-Learn development
team.  This functionality is progressing quickly but is in a state of rapid
change.*

Reimplement Scalable Algorithms with Dask Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some machine learning algorithms are easy to write down as Numpy algorithms.
In these cases we can replace Numpy arrays with Dask arrays to achieve scalable
algorithms easily.  This is employed for :doc:`linear models <glm>`, :doc:`pre-processing <preprocessing>`, and :doc:`clustering <clustering>`.

.. code-block:: python

   from dask_ml.preprocessing import Categorizer, DummyEncoder
   from dask_ml.linear_model import LogisticRegression

   lr = LogisticRegression()
   lr.fit(data, labels)

Partner with other distributed libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other machine learning libraries like XGBoost and TensorFlow already have
distributed solutions that work quite well.  Dask-ML makes no attempt to
re-implement these systems.  Instead, Dask-ML makes it easy to use normal Dask
workflows to prepare and set up data, then it deploys XGBoost or Tensorflow
*alongside* Dask, and hands the data over.

.. code-block:: python

   from dask_ml.xgboost import XGBRegressor

   est = XGBRegressor(...)
   est.fit(train, train_labels)

See :doc:`Dask-ML + XGBoost <xgboost>` or :doc:`Dask-ML + TensorFlow
<tensorflow>` documentation for more information.


Scikit-Learn API
----------------

In all cases Dask-ML endeavors to provide a single unified interface around the
familiar NumPy, Pandas, and Scikit-Learn APIs.  Users familiar with
Scikit-Learn should feel at home with Dask-ML.

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
   cross_validation.rst
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
   roadmap.rst
   history.rst

.. _Dask: https://dask.pydata.org/
.. _Scikit-Learn: http://scikit-learn.org/
