dask-searchcv
=============

Tools for performing hyperparameter optimization of Scikit-Learn models using
Dask.

Introduction
------------

This library provides implementations of Scikit-Learn's ``GridSearchCV`` and
``RandomizedSearchCV``. They implement many (but not all) of the same
parameters, and should be a drop-in replacement for the subset that they do
implement. For certain problems, these implementations can be more efficient
than those in Scikit-Learn, as they can avoid expensive repeated computations.

Highlights
----------

- Drop-in replacement for Scikit-Learn's ``GridSearchCV`` and
  ``RandomizedSearchCV``.

- Hyperparameter optimization can be done in parallel using threads, processes,
  or distributed across a cluster.

- Works well with Dask collections. Dask arrays, dataframes, and delayed can be
  passed to ``fit``.

- Candidate estimators with identical parameters and inputs will only be fit
  once. For meta-estimators such as ``Pipeline`` this can be significantly more
  efficient as it can avoid expensive repeated computations.

Example
-------

.. code-block:: python

    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    import dask_searchcv as dcv
    import numpy as np

    digits = load_digits()

    param_space = {'C': np.logspace(-4, 4, 9),
                   'gamma': np.logspace(-4, 4, 9),
                   'class_weight': [None, 'balanced']}

    model = SVC(kernel='rbf')
    search = dcv.GridSearchCV(model, param_space, cv=3)

    search.fit(digits.data, digits.target)

Index
-----

.. toctree::

    api
