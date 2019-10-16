Estimators
==========

The :mod:`estimators` module offers a scikit-learn compatible API for
specifying your model and hyper-parameters, and fitting your model to data.

.. code-block:: python

   >>> from dask_glm.estimators import LogisticRegression
   >>> from dask_glm.datasets import make_classification
   >>> X, y = make_classification()
   >>> lr = LogisticRegression()
   >>> lr.fit(X, y)
   >>> lr
   LogisticRegression(abstol=0.0001, fit_intercept=True, lamduh=1.0,
             max_iter=100, over_relax=1, regularizer='l2', reltol=0.01, rho=1,
             solver='admm', tol=0.0001)


All of the estimators follow a similar API. They can be instantiated with
a set of parameters that control the fit, including whether to add an intercept,
which solver to use, how to regularize the inputs, and various optimization
parameters.

Given an instantiated estimator, you pass the data to the ``.fit`` method.
It takes an ``X``, the feature matrix or exogenous data, and a ``y`` the
target or endogenous data. Each of these can be a NumPy or dask array.

With a fit model, you can make new predictions using the ``.predict`` method,
and can score known observations with the ``.score`` method.

.. code-block:: python

   >>> lr.predict(X).compute()
   array([False, False, False, True, ... True, False, True, True], dtype=bool)

See the :ref:`api-reference` for more.
