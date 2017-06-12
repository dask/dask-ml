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

For more information, see `this blogpost
<http://jcrist.github.io/introducing-dask-searchcv.html>`__.

Highlights
----------

- :ref:`Drop-in replacement <drop-in-replacement>` for Scikit-Learn's
  ``GridSearchCV`` and ``RandomizedSearchCV``.

- :ref:`Flexible Backends <flexible-backends>`. Hyperparameter
  optimization can be done in parallel using threads, processes, or distributed
  across a cluster.

- :ref:`Works well with Dask collections <works-with-dask-collections>`. Dask
  arrays, dataframes, and delayed can be passed to ``fit``.

- :ref:`Avoid repeated work <avoid-repeated-work>`. Candidate estimators with
  identical parameters and inputs will only be fit once. For
  composite-estimators such as ``Pipeline`` this can be significantly more
  efficient as it can avoid expensive repeated computations.

Install
-------

Dask-searchcv is available via ``conda`` or ``pip``:

.. code-block:: bash

   # Install with conda
   $ conda install dask-searchcv -c conda-forge

   # Install with pip
   $ pip install dask-searchcv


Walkthrough
-----------

.. _drop-in-replacement:

Drop-In Replacement
^^^^^^^^^^^^^^^^^^^

Dask-searchcv provides (almost) drop-in replacements for Scikit-Learn's
``GridSearchCV`` and ``RandomizedSearchCV``. With the exception of a few
keyword arguments, the api's are exactly the same, and often only an import
change is necessary:

.. code-block:: python
    :emphasize-lines: 4,5

    from sklearn.datasets import load_digits
    from sklearn.svm import SVC

    # Fit with dask-searchcv
    from dask_searchcv import GridSearchCV

    param_space = {'C': [1e-4, 1, 1e4],
                   'gamma': [1e-3, 1, 1e3],
                   'class_weight': [None, 'balanced']}

    model = SVC(kernel='rbf')

    digits = load_digits()

    search = GridSearchCV(model, param_space, cv=3)
    search.fit(digits.data, digits.target)

.. raw:: html

    <!-- This is for cycling the comment/import lines in the above codeblock -->
    <script type="text/javascript">
        var text1 = ["# Fit with scikit-learn", "# Fit with dask-searchcv"];
        var text2 = ["sklearn.model_selection", "dask_searchcv"];
        var counter = 0;
        function find(cls, target) {
            elements = document.getElementsByClassName(cls)
            for (i = 0; i < elements.length; i++) {
                if (elements[i].innerHTML == target) { return elements[i]; }
            }
        }
        var elem1 = find("c1", text1[1]);
        var elem2 = find("nn", text2[1]);
        setInterval(change, 2000);
        function change() {
            elem1.innerHTML = text1[counter];
            elem2.innerHTML = text2[counter];
            counter++;
            if(counter >= 2) { counter = 0; }
        }
    </script>


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
``sklearn.pipeline.FeatureUnion``, dask-searchcv will avoid fitting the same
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
   :alt: "dask-searchcv grid-search directed acyclic graph"
   :align: center


Looking closely, you can see that the Scikit-Learn version ends up fitting
earlier steps in the pipeline multiple times with the same parameters and data.
Due to the increased flexibility of Dask over Joblib, we're able to merge these
tasks in the graph and only perform the fit step once for any
parameter/data/estimator combination. For pipelines that have relatively
expensive early steps, this can be a big win when performing a grid search.


Index
-----

.. toctree::

    api
