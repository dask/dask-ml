Contributing
============

Thanks for helping to build ``dask-ml``!

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~

Make a fork of the `repo <https://github.com/dask/dask-ml>`__ and clone that

.. code-block:: none

   git clone https://github.com/<your-github-username>/dask-ml
   cd dask-ml

You may want to add ``https://github.com/dask/dask-ml`` as an upstream.

.. code-block::none

   git remote add upstream https://github.com/dask/dask-ml

Creating an environment
~~~~~~~~~~~~~~~~~~~~~~~

We have an conda ``environment.yaml`` with all the dependencies. If you're using
conda you can

.. code-block:: none

   conda env create -f ci/environment.yml --name=dask-ml-dev

If you're using pip, check out the ``setup.py`` for the required and optional
dependencies. You'll at lest need the build dependencies of NumPy, setuptools,
setuptools_scm, and Cython.

Building dask-ml
~~~~~~~~~~~~~~~~

The library has some C-extensions, so installing is a bit more complicated than
normal. If you have a compiler and everything is setup correctly, you should be
able to install Cython and all the required dependencies.

From within the repository:

.. code-block:: none

   python setup.py build_ext --inplace

And then

.. code-block:: none

   python -m pip install -e .[dev]

If you have any trouble with the build step, please open an issue.

Conventions
~~~~~~~~~~~

For the most part, we follow scikit-learn's API design. If you're implementing
a new estimator, it will ideally pass scikit-learn's `estimator check`_.

We have some additional decisions to make in the dask context. Ideally

1. All attributes learned during ``.fit`` should be *concrete*, i.e. they should
   not be dask collections.
2. To the extent possible, transformers should support

   * ``numpy.ndarray``
   * ``pandas.DataFrame``
   * ``dask.Array``
   * ``dask.DataFrame``

3. If possible, transformers should accept a ``columns`` keyword to limit the
   transformation to just those columns, while passing through other columns
   untouched. ``inverse_transform`` should behave similarly (ignoring other
   columns) so that ``inverse_transform(transform(X))`` equals ``X``.
4. Methods returning arrays (like ``.transform``, ``.predict``), should return
   the same type as the input. So if a ``dask.array`` is passed in, a
   ``dask.array`` with the same chunks should be returned.

.. _estimator check: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
