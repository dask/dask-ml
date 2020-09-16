.. _dask-ml:

=======
Dask-ML
=======

Dask-ML provides scalable machine learning in Python using Dask_ alongside
popular machine learning libraries like Scikit-Learn_, XGBoost, and others.

You can try Dask-ML on a small cloud instance by clicking the following button:

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/dask/dask-examples/master?filepath=machine-learning.ipynb

Dimensions of Scale
-------------------

People may run into scaling challenges along a couple dimensions, and Dask-ML
offers tools for addressing each.

.. image:: images/dimensions_of_scale.svg


Challenge 1: Scaling Model Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first kind of scaling challenge comes when from your models growing so
large or complex that it affects your workflow (shown along the vertical axis
above). Under this scaling challenge tasks like model training, prediction, or
evaluation steps will (eventually) complete, they just take too long. You've
become compute bound.

To address these challenges you'd continue to use the collections you know and
love (like the NumPy ``ndarray``, pandas ``DataFrame``, or XGBoost ``DMatrix``)
and use a Dask Cluster to parallelize the workload on many machines. The
parallelization can occur through one of our integrations (like Dask's
:ref:`joblib backend <joblib>` to parallelize Scikit-Learn directly) or one of
Dask-ML's estimators (like our :ref:`hyper-parameter optimizers
<hyper-parameter-search>`).

Challenge 2: Scaling Data Size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second type of scaling challenge people face is when their datasets grow
larger than RAM (shown along the horizontal axis above). Under this scaling
challenge, even loading the data into NumPy or pandas becomes impossible.

To address these challenges, you'd use Dask's one of Dask's `high-level
collections <https://docs.dask.org/en/latest/user-interfaces.html>`__ like
(`Dask Array <https://docs.dask.org/en/latest/array.html>`__, `Dask DataFrame
<https://docs.dask.org/en/latest/dataframe.html>`__ or `Dask Bag
<https://docs.dask.org/en/latest/bag.html>`__) combined with one of Dask-ML's
estimators that are designed to work with Dask collections. For example you
might use Dask Array and one of our preprocessing estimators in
:mod:`dask_ml.preprocessing`, or one of our ensemble methods in
:mod:`dask_ml.ensemble`.

Not Everyone needs Scalable Machine Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's worth emphasizing that not everyone needs scalable machine learning. Tools
like sampling can be effective. Always plot your `learning curve
<https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html>`__.


Scikit-Learn API
----------------

In all cases Dask-ML endeavors to provide a single unified interface around the
familiar NumPy, Pandas, and Scikit-Learn APIs.  Users familiar with
Scikit-Learn should feel at home with Dask-ML.

Partner with other distributed libraries
----------------------------------------

Other machine learning libraries like XGBoost already have
distributed solutions that work quite well.  Dask-ML makes no attempt to
re-implement these systems.  Instead, Dask-ML makes it easy to use normal Dask
workflows to prepare and set up data, then it deploys XGBoost
*alongside* Dask, and hands the data over.

.. code-block:: python

   from dask_ml.xgboost import XGBRegressor

   est = XGBRegressor(...)
   est.fit(train, train_labels)

See :doc:`Dask-ML + XGBoost <xgboost>` for more information.


.. toctree:: :maxdepth: 2 :hidden: :caption: Get Started

   install.rst
   examples.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Use

   preprocessing.rst
   cross_validation.rst
   hyper-parameter-search.rst
   compose.rst
   glm.rst
   meta-estimators.rst
   incremental.rst
   clustering.rst
   modules/api.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Integration

   joblib.rst
   xgboost.rst
   pytorch.rst
   keras.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Develop

   changelog.rst
   contributing.rst
   roadmap.rst
   history.rst

.. _Dask: https://dask.org/
.. _Scikit-Learn: http://scikit-learn.org/
