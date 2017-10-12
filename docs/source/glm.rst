Generalized Linear Models
=========================

.. currentmodule:: daskml.linear_model

.. autosummary::
   LinearRegression
   LogisticRegression
   PoissonRegression

`dask-glm`_ is a library for fitting Generalized Linear Models that's built on
top of dask. Typically, you'd use it like a regular scikit-learn estimator:
instantiate an estimator and call ``estimator.fit(X, y)`` with dask arrays.
See :ref:`examples/dask-glm.ipynb` for an example.

The ``dask-glm`` estimators are available from the ``daskml.linear_model``
namespace.

.. _dask-glm: http://dask-glm.readthedocs.io/
