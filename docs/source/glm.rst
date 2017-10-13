Generalized Linear Models
=========================

.. currentmodule:: dask_ml.linear_model

.. autosummary::
   LinearRegression
   LogisticRegression
   PoissonRegression

Generalized linear models are a broad class of commonly used models.  These
implementations scale well out to large datasets either on a single machine or
distributed cluster.  They can be powered by a variety of optimization
algorithms and use a variety of regularizers.

These follow the scikit-learn estimator API, and so can be dropped into
existing routines like grid search and pipelines, but are implemented
externally with new, scalable algorithms and so can consume distributed dask
arrays and dataframes rather than just single-machine NumPy and Pandas arrays
and dataframes.

Example
-------

.. code-block:: python

   >>> from dask_glm.estimators import LogisticRegression
   >>> from dask_glm.datasets import make_classification
   >>> X, y = make_classification()
   >>> lr = LogisticRegression()
   >>> lr.fit(X, y)
   >>> lr
   LogisticRegression(abstol=0.0001, fit_intercept=True, lamduh=1.0,
                      max_iter=100, over_relax=1, regularizer='l2', reltol=0.01,
                      rho=1, solver='admm', tol=0.0001)


Algorithms
----------

.. currentmodule:: dask_glm.algorithms

.. autosummary::
   admm
   gradient_descent
   lbfgs
   newton
   proximal_grad


Regularizers
------------

.. currentmodule:: dask_glm.regularizers

.. autosummary::
   ElasticNet
   L1
   L2
   Regularizer
