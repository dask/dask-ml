Contributing
============

Thanks for helping to build ``dask-ml``!

Cloning the Repository
~~~~~~~~~~~~~~~~~~~~~~

Make a fork of the `dask-ml repo <https://github.com/dask/dask-ml>`__ and clone
the fork

.. code-block:: none

   git clone https://github.com/<your-github-username>/dask-ml
   cd dask-ml

You may want to add ``https://github.com/dask/dask-ml`` as an upstream remote
repository.

.. code-block:: none

   git remote add upstream https://github.com/dask/dask-ml

Creating an environment
~~~~~~~~~~~~~~~~~~~~~~~

We have conda environment YAML files with all the necessary dependencies
in the ``ci`` directory. If you're using conda you can

.. code-block:: none

   conda env create -f ci/environment-3.6.yml --name=dask-ml-dev

to create a conda environment and install all the dependencies. Note there is
also a ``ci/environment-2.7.yml`` file if you need to use Python 2.7.

If you're using pip, you can view the list of all the required and optional
dependencies within ``setup.py`` (see the ``install_requires`` field for
required dependencies and ``extras_require`` for optional dependencies). You'll
at least need the build dependencies of NumPy, setuptools, setuptools_scm, and
Cython.

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

   python -m pip install -e ".[dev]"

If you have any trouble with the build step, please open an issue in the
`dask-ml issue tracker <https://github.com/dask/dask-ml/issues>`_.

Running tests
~~~~~~~~~~~~~

Dask-ml uses `py.test <https://docs.pytest.org/en/latest/>`_ for testing. You
can run tests from the main dask-ml directory as follows:

.. code-block:: none

    py.test tests

Alternatively you may choose to run only a subset of the full test suite. For
example to test only the preprocessing submodule we would run tests as follows:

.. code-block:: none

    py.test tests/preprocessing


In addition to running tests, dask-ml verifies code style uniformity with the
``flake8`` tool:

.. code-block:: none

    pip install flake8
    flake8


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
