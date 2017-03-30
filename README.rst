dask-searchcv
=============

|Travis Status|

Tools for performing hyperparameter search with Scikit-Learn and Dask.

This library provides implementations of Scikit-Learn's ``GridSearchCV`` and
``RandomizedGridSearchCV``. They implement many (but not all) of the same
parameters, and should be a drop-in replacement for the subset that they do
implement. For certain problems, these implementations can be more efficient
than those in scikit-learn, as they can avoid repeating expensive repeated
computations.

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


.. |Travis Status| image:: https://travis-ci.org/dask/dask-searchcv.svg?branch=master
   :target: https://travis-ci.org/dask/dask-searchcv
