dask-searchcv
=============

|Travis Status| |Doc Status| |Conda Badge| |PyPI Badge|

Tools for performing hyperparameter search with
`Scikit-Learn <http://scikit-learn.org>`_ and `Dask <http://dask.pydata.org>`_.

Highlights
----------

- Drop-in replacement for Scikit-Learn's ``GridSearchCV`` and
  ``RandomizedSearchCV``.

- Hyperparameter optimization can be done in parallel using threads, processes,
  or distributed across a cluster.

- Works well with Dask collections. Dask arrays, dataframes, and delayed can be
  passed to ``fit``.

- Candidate estimators with identical parameters and inputs will only be fit
  once. For composite-estimators such as ``Pipeline`` this can be significantly
  more efficient as it can avoid expensive repeated computations.


For more information, check out the `documentation <http://dask-searchcv.readthedocs.io>`_.


Install
-------

Dask-searchcv is available via ``conda`` or ``pip``:

::

   # Install with conda
   $ conda install dask-searchcv -c conda-forge

   # Install with pip
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
.. |Conda Badge| image:: https://anaconda.org/conda-forge/dask-searchcv/badges/version.svg
   :target: https://anaconda.org/conda-forge/dask-searchcv
