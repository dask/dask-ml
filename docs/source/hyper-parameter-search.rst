Hyper Parameter Search
======================

*Tools for performing hyperparameter optimization of Scikit-Learn API-compatible models using Dask*.

Issues in hyper-parameter searches
----------------------------------
Two scenarios can occur during hyper-parameter optimization. The
hyper-parameter search can be both

1. compute constrained
2. memory constrained

These issues are independent and both can happen the same time. Being memory
constrained has to do with dataset size, and being compute constrained has to
do with estimator complexity and number of possible hyper-parameter
combinations.

An example of being compute constrained is with almost any neural network or
deep learning framework. An example of being memory constrained is when the
dataset doesn't fit in RAM. Dask-ML covers all 4 combinations of these two
constraints.

Neither compute nor memory constrained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Scikit-learn handles this case.

.. autosummary::
   sklearn.model_selection.GridSearchCV
   sklearn.model_selection.RandomizedSearchCV

Compute constrained, but not memory constrained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More detail in :ref:`hyperparameter.drop-in`

.. autosummary::
   dask_ml.model_selection.GridSearchCV
   dask_ml.model_selection.RandomizedSearchCV

Both :class:`~dask_ml.model_selection.GridSearchCV` and
:class:`~dask_ml.model_selection.RandomizedSearchCV` are especially good for
pipelines because they avoid repeated work. They are drop in replacement
for the Scikit-learn versions.

Memory constrained, but not compute constrained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More detail in :ref:`hyperparameter.incremental`

.. autosummary::
   dask_ml.model_selection.IncrementalSearchCV

By default :class:`~dask_ml.model_selection.IncrementalSearchCV` mirrors
:class:`~dask_ml.model_selection.RandomizedSearchCV` and
:class:`~dask_ml.model_selection.GridSearchCV` but calls `partial_fit` instead
of `fit`. This means that it can scale to much larger datasets, including ones
that don't fit in the memory of a single machine.

Memory constrained and compute constrained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more detail, see :ref:`hyperparameter.hyperband`.

.. autosummary::
   dask_ml.model_selection.HyperbandSearchCV
   dask_ml.model_selection.IncrementalSearchCV
   dask_ml.model_selection.SuccessiveHalvingSearchCV

These searches can
reduce time to solution by (cleverly) deciding which parameters to evaluate.
These searches `adapt` to history to decide which parameters to continue
evaluating and are called "`adaptive` model selection algorithms".

This can drastically reduce the computation required and make the problem many
times simpler. These classes require that the estimator implement ``partial_fit``.

.. _hyperparameter.drop-in:

Drop-In Replacements for Scikit-Learn
-------------------------------------

Dask-ML implements drop-in replacements for
:class:`~sklearn.model_selection.GridSearchCV` and
:class:`~sklearn.model_selection.RandomizedSearchCV`.

.. autosummary::
   dask_ml.model_selection.GridSearchCV
   dask_ml.model_selection.RandomizedSearchCV

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

.. autosummary::
   dask_ml.model_selection.IncrementalSearchCV

.. note::

   These estimators require the optional ``distributed`` library.

These are make repeated calls to the ``partial_fit`` method of the estimator.
Naturally, these classes determine when to stop calling ``partial_fit`` by
`adapting to previous calls`. The most basic level of this is to stop training
if the score doens't improve, which ``IncrementalSearchCV`` does. For more
advanced methods, see :ref:`hyperparameter.hyperband`.


Basic use
^^^^^^^^^

.. ipython:: python

    from dask.distributed import Client
    client = Client()
    import numpy as np
    from dask_ml.datasets import make_classification
    # X, y = make_classification(n_samples=5000000, n_features=20,
    #                           chunks=100000, random_state=0)
    X, y = make_classification(chunks=20, random_state=0)

Our underlying estimator is an ``SGDClassifier``. We specify a few parameters
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

    from dask_ml.model_selection import HyperbandSearchCV

    search = HyperbandSearchCV(model, params, 9, random_state=0)
    _ = search.fit(X, y, classes=[0, 1])
    search.best_score_
    search.best_params_

Note that when you do post-fit tasks like ``search.score``, the underlying
estimator's score method is used. If that is unable to handle a
larger-than-memory Dask Array, you'll exhaust your machines memory. If you plan
to use post-estimation features like scoring or prediction, we recommend using
:class:`dask_ml.wrappers.ParallelPostFit`.

.. ipython:: python

   from dask_ml.wrappers import ParallelPostFit
   params = {'estimator__alpha': np.logspace(-2, 1, num=1000)}
   model = ParallelPostFit(SGDClassifier(tol=1e-3, random_state=0))
   search = HyperbandSearchCV(model, params, 9, random_state=0)
   _ = search.fit(X, y, classes=[0, 1])
   search.score(X, y)

Note that the parameter names include the ``estimator__`` prefix,
as we're tuning the hyperparameters of the ``SGDClassifier`` that's
underlying the ``ParallelPostFit``.

.. _hyperparameter.hyperband:

Adaptive hyperparameter search
------------------------------

.. autosummary::
   dask_ml.model_selection.HyperbandSearchCV
   dask_ml.model_selection.IncrementalSearchCV
   dask_ml.model_selection.SuccessiveHalvingSearchCV

We most recommend use of :class:`~dask_ml.model_selection.HyperbandSearchCV`.
The two other implementations,
:class:`~dask_ml.model_selection.IncrementalSearchCV` and
:class:`~dask_ml.model_selection.SuccessiveHalvingSearchCV` are inspired and
used by :class:`~dask_ml.model_selection.HyperbandSearchCV` respectively. We
recommend it for reasons detailed in :ref:`hyperparameter.hyperband`.

HyperbandSearchCV offers two benefits:

1. It finds better models quicker
2. It requires only two inputs

High performing models
^^^^^^^^^^^^^^^^^^^^^^

Hyperband requires minimal computation because it has guarantees on
finding the best set of parameters possible with a given number of
``partial_fit`` calls [HY16]. [#qual]_ This is possible because Hyperband
balances two extremes:

* when only training time is important
    * i.e., when the hyper-parameters don't influence the output at all)
* when training time doesn't matter at all
    * i.e., when the hyper-parameters exactly determine the output

Parameters
^^^^^^^^^^

:class:`~dask_ml.model_selection.HyperbandSearchCV` requires knowing two items:

* how many examples to pass to the estimator
* how many parameters to initially evaluate

Hyperband's required parameters fall out pretty naturally and simply from these
two items, which is detailed in
:class:`~dask_ml.model_selection.HyperbandSearchCV`'s documentation.

.. [#qual] More accurately, Hyperband will find "close" to the best model in expected value with high probability, where "close" is "within log factors of the lower bound".
.. [HY16] "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" by Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh and Ameet Talwalkar. https://arxiv.org/abs/1603.06560
