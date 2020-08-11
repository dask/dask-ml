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
in the ``ci`` directory.

.. code-block:: none

   conda env create -f ci/environment-latest.yaml --name=dask-ml-dev

to create a conda environment and install all the dependencies.

If you're using pip, you can view the list of all the required and optional
dependencies within ``setup.py`` (see the ``install_requires`` field for
required dependencies and ``extras_require`` for optional dependencies).

Building dask-ml
~~~~~~~~~~~~~~~~

Dask-ML is a pure-python repository. Development installation should be as simple as
cloning the repository and running the following in the cloned directory:

``pip``::

  python -m pip install -e ".[dev]"

``conda``::
   
  conda activate dask-ml-dev
  python -m pip install --no-deps -e .

If you have any trouble, please open an issue on the
`dask-ml issue tracker <https://github.com/dask/dask-ml/issues>`_.

Style
~~~~~

Dask-ML uses `black <http://black.readthedocs.io/en/stable/>`_ for formatting
and `flake8 <http://flake8.pycqa.org/en/latest/>`_ for linting. If you installed
dask-ml with ``python -m pip install -e ".[dev]"`` these tools will already be
installed.

.. code-block:: none

    black .
    flake8
    isort -rc dask_ml tests

You may wish to setup a
`pre-commit hook <https://black.readthedocs.io/en/stable/version_control_integration.html>`_
to run black when you commit changes.

Running tests
~~~~~~~~~~~~~

Dask-ml uses `py.test <https://docs.pytest.org/en/latest/>`_ for testing. You
can run tests from the main dask-ml directory as follows:

.. code-block:: none

    pytest tests

Alternatively you may choose to run only a subset of the full test suite. For
example to test only the preprocessing submodule we would run tests as follows:

.. code-block:: none

    pytest tests/preprocessing

Coverage
~~~~~~~~

If your Pull Request decreases the lines of code covered, the CI may fail.
Sometimes this is OK, and a maintainer will merge it anyway. To check the coverage locally,
use

.. code-block:: none

   pytest --cov --cov-report=html

You can still use all the usual pytest command-line options in addition to those.

Pre-Commit Hooks
~~~~~~~~~~~~~~~~

Install and build the `pre commit <https://github.com/pre-commit/pre-commit>`_ tool as:

.. code-block:: none

    python -m pip install pre-commit
    pre-commit install

to install a few plugins like black, isort, and flake8. These tools will automatically
be run on each commit. You can skip the checks with ``git commit --no-verify``.

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

.. _estimator check: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator

Documentation
~~~~~~~~~~~~~

We use `numpydoc <http://numpydoc.readthedocs.io/en/latest/format.html>`_ for our docstrings.

Building the docs is possible with

.. code-block:: none

   $ conda env create -f ci/environment-docs.yaml --name=dask-ml-dev-docs
   $ conda activate dask-ml-dev-docs
   $ python -m pip install -e .
   $ cd docs
   $ make html

Examples are written as Jupyter notebooks with their output stripped, either
manually or using `nbstripout <https://github.com/kynan/nbstripout>`_. We want
examples to be runnable on binder so they should be small, but include
instructions for how to scale up to larger problems.

The source for most examples is maintained in the `dask-examples
<https://github.com/dask/dask-examples>`_ repository. Updates should be made
there, and they're automatically included as part of the Dask-ML documentation
build process.

When adding an example for new feature that's only available in master, the
notebook should be first included in Dask-ML repository under
``docs/source/examples/``. These examples will be moved to
``dask/dask-examples`` as part of the Dask-ML release process.
