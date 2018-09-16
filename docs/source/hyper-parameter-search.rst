Hyper Parameter Search
======================

*Tools for performing hyperparameter optimization of Scikit-Learn API-compatible models using Dask*.

There are two kinds of hyperparameter optimization estimators
in Dask-ML. The appropriate one to use depends on the size of your dataset and
whether the underlying estimator implements the `partial_fit` method.

If your dataset is relatively small or the underlying estimator doesn't implement
``partial_fit``, you can use :class:`dask_ml.model_selection.GridSearchCV` or
:class:`dask_ml.model_selection.RandomizedSearchCV`.
These are drop-in replacements for their scikit-learn counterparts, that should offer better performance and handling of Dask Arrays and DataFrames.
The underlying estimator will need to be able to train on each cross-validation split of the data.
See :ref:`hyperparameter.drop-in` for more.

If your data is large and the underlying estimator implements ``partial_fit``, you can
Dask-ML's :ref:`*incremental* hyperparameter optimizers <hyperparameter.incremental>`.

.. _hyperparameter.drop-in:

Drop-In Replacements for Scikit-Learn
-------------------------------------

Dask-ML implements GridSearchCV and RandomizedSearchCV.

.. autosummary::
   sklearn.model_selection.GridSearchCV
   dask_ml.model_selection.GridSearchCV
   sklearn.model_selection.RandomizedSearchCV
   dask_ml.model_selection.RandomizedSearchCV
   dask_ml.model_selection.HyperbandCV

The varians in Dask-ML implement many (but not all) of the same parameters,
and should be a drop-in replacement for the subset that they do implement.
In that case, why use Dask-ML's versions?

- :ref:`Flexible Backends <flexible-backends>`: Hyperparameter
  optimization can be done in parallel using threads, processes, or distributed
  across a cluster.

- :ref:`Works well with Dask collections <works-with-dask-collections>`. Dask
  arrays, dataframes, and delayed can be passed to ``fit``.

- :ref:`Avoid repeated work <avoid-repeated-work>`. Candidate estimators with
  identical parameters and inputs will only be fit once. For
  composite-estimators such as ``Pipeline`` this can be significantly more
  efficient as it can avoid expensive repeated computations.

- :ref:`Adaptive algorithms <adaptive>` like Hyperband, which
    - uses previous estimator evaluation to determine which estimator to
      evaluate next.
    - are (fairly) well suited for Dask's architecture.

Both scikit-learn's and Dask-ML's model selection meta-estimators can be used
with Dask's :ref:`joblib backend <joblib>`.

.. _flexible-backends:

Flexible Backends
^^^^^^^^^^^^^^^^^

Dask-searchcv can use any of the dask schedulers. By default the threaded
scheduler is used, but this can easily be swapped out for the multiprocessing
or distributed scheduler:

.. code-block:: python

    # Distribute grid-search across a cluster
    from dask.distributed import Client
    scheduler_address = '127.0.0.1:8786'
    client = Client(scheduler_address)

    search.fit(digits.data, digits.target)


.. _works-with-dask-collections:

Works Well With Dask Collections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dask collections such as ``dask.array``, ``dask.dataframe`` and
``dask.delayed`` can be passed to ``fit``. This means you can use dask to do
your data loading and preprocessing as well, allowing for a clean workflow.
This also allows you to work with remote data on a cluster without ever having
to pull it locally to your computer:

.. code-block:: python

    import dask.dataframe as dd

    # Load data from s3
    df = dd.read_csv('s3://bucket-name/my-data-*.csv')

    # Do some preprocessing steps
    df['x2'] = df.x - df.x.mean()
    # ...

    # Pass to fit without ever leaving the cluster
    search.fit(df[['x', 'x2']], df['y'])


.. _avoid-repeated-work:

Avoid Repeated Work
^^^^^^^^^^^^^^^^^^^

When searching over composite estimators like ``sklearn.pipeline.Pipeline`` or
``sklearn.pipeline.FeatureUnion``, Dask-ML will avoid fitting the same
estimator + parameter + data combination more than once. For pipelines with
expensive early steps this can be faster, as repeated work is avoided.

For example, given the following 3-stage pipeline and grid (modified from `this
scikit-learn example
<http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html>`__).

.. code-block:: python

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier())])

    grid = {'vect__ngram_range': [(1, 1)],
            'tfidf__norm': ['l1', 'l2'],
            'clf__alpha': [1e-3, 1e-4, 1e-5]}

the Scikit-Learn grid-search implementation looks something like (simplified):

.. code-block:: python

	scores = []
	for ngram_range in parameters['vect__ngram_range']:
		for norm in parameters['tfidf__norm']:
			for alpha in parameters['clf__alpha']:
				vect = CountVectorizer(ngram_range=ngram_range)
				X2 = vect.fit_transform(X, y)
				tfidf = TfidfTransformer(norm=norm)
				X3 = tfidf.fit_transform(X2, y)
				clf = SGDClassifier(alpha=alpha)
				clf.fit(X3, y)
				scores.append(clf.score(X3, y))
	best = choose_best_parameters(scores, parameters)


As a directed acyclic graph, this might look like:

.. figure:: images/unmerged_grid_search_graph.svg
   :alt: "scikit-learn grid-search directed acyclic graph"
   :align: center


In contrast, the dask version looks more like:

.. code-block:: python

	scores = []
	for ngram_range in parameters['vect__ngram_range']:
		vect = CountVectorizer(ngram_range=ngram_range)
		X2 = vect.fit_transform(X, y)
		for norm in parameters['tfidf__norm']:
			tfidf = TfidfTransformer(norm=norm)
			X3 = tfidf.fit_transform(X2, y)
			for alpha in parameters['clf__alpha']:
				clf = SGDClassifier(alpha=alpha)
				clf.fit(X3, y)
				scores.append(clf.score(X3, y))
	best = choose_best_parameters(scores, parameters)


With a corresponding directed acyclic graph:

.. figure:: images/merged_grid_search_graph.svg
   :alt: "Dask-ML grid-search directed acyclic graph"
   :align: center


Looking closely, you can see that the Scikit-Learn version ends up fitting
earlier steps in the pipeline multiple times with the same parameters and data.
Due to the increased flexibility of Dask over Joblib, we're able to merge these
tasks in the graph and only perform the fit step once for any
parameter/data/estimator combination. For pipelines that have relatively
expensive early steps, this can be a big win when performing a grid search.

.. _hyperparameter.incremental:


Incremental Hyperparameter Optimization
---------------------------------------

The second category of hyperparameter optimization uses *incremental*
hyperparameter optimization. These should be used when your full dataset doesn't
fit in memory on a single machine.

.. autosummary::
   dask_ml.model_selection.IncrementalSearchCV

Broadly speaking, incremental optimization starts with a batch of models (underlying
estimators and hyperparameter combinations) and repeatedly calls the underlying estimator's
``partial_fit`` method with batches of data.

.. note::

   These estimators require the optional ``distributed`` library.

Here's an example training on a "large" dataset (a Dask array) with the
``IncrementalSearchCV``.

.. ipython:: python

    from dask.distributed import Client
    client = Client()
    import numpy as np
    from dask_ml.datasets import make_classification
    X, y = make_classification(n_samples=5000000, n_features=20,
                               chunks=100000, random_state=0)

Our underlying estimator is an SGDClassifier. We specify a few parameters
common to each clone of the estimator:

.. ipython:: python

    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

We also define the distribution of parameters from which we will sample:

.. ipython:: python

    params = {'alpha': np.logspace(-2, 1, num=1000),
              'l1_ratio': np.linspace(0, 1, num=1000),
              'average': [True, False]}


Finally we create many random models in this parameter space and
train-and-score them until we find the best one.

.. ipython:: python

    from dask_ml.model_selection import IncrementalSearchCV

    search = IncrementalSearchCV(model, params, random_state=0)
    search.fit(X, y, classes=[0, 1])

Note that when you do post-fit tasks like ``search.score``, the underlying
estimator's score method is used. If that is unable to handle a
larger-than-memory Dask Array, you'll exhaust your machines memory. If you plan
to use post-estimation features like scoring or prediction, we recommend using
:class:`dask_ml.wrappers.ParallelPostFit`.

.. ipython:: python

   from dask_ml.wrappers import ParallelPostFit

   params = {'estimator__alpha': np.logspace(-2, 1, num=1000),
             'estimator__l1_ratio': np.linspace(0, 1, num=1000),
             'estimator__average': [True, False]}

   model = ParallelPostFit(SGDClassifier(tol=1e-3,
                                         penalty="elasticnet",
                                         random_state=0))
   search = IncrementalSearchCV(model, params, random_state=0)
   search.fit(X, y, classes=[0, 1])
   search.score(X, y)

Note that the parameter names include the ``estimator__`` prefix,
as we're tuning the hyperparameters of the ``SGDClassifier`` that's
underlying the ``ParallelPostFit``.
