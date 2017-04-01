dask-searchcv
=============

|Travis Status| |Doc Status| |PyPI Badge|

Tools for performing hyperparameter search with
`Scikit-Learn <http://scikit-learn.org>`_ and `Dask <http://dask.pydata.org>`_.

This library provides implementations of Scikit-Learn's ``GridSearchCV`` and
``RandomizedSearchCV``. They implement many (but not all) of the same
parameters, and should be a drop-in replacement for the subset that they do
implement. For certain problems, these implementations can be more efficient
than those in Scikit-Learn, as they can avoid expensive repeated computations.

For more information, check out the `documentation <http://dask-searchcv.readthedocs.io>`_.

Install
-------

Dask-searchcv is available via ``pip``:

::

   $ pip install dask-searchcv


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


.. |Travis Status| image:: https://travis-ci.org/dask/dask-searchcv.svg?branch=master
   :target: https://travis-ci.org/dask/dask-searchcv
.. |Doc Status| image:: http://readthedocs.org/projects/dask-searchcv/badge/?version=latest
   :target: http://dask-searchcv.readthedocs.io/en/latest/index.html
   :alt: Documentation Status
.. |PyPI Badge| image:: https://img.shields.io/pypi/v/dask-searchcv.svg
   :target: https://pypi.python.org/pypi/dask-searchcv
