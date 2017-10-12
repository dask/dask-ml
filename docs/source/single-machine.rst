.. _single-machine:

==============
Single Machine
==============

First, we need to recognize that most machine learning frameworks already use
parallelism in quite a few places. For example, in scikit-learn anywhere you see
an ``n_jobs`` parameter, scikit-learn will already be using some parallelism.
That said, dask can still improve performance through its sophisticated caching
when fitting a a :class:`sklearn.pipeline.Pipeline`.

Pipelines
---------

First, some non-dask-related background:
A :class:`sklearn.pipeline.Pipeline` makes it possible to define the entire modeling
process, from raw data to fit estimator, in a single python object. You can
create a pipeline with :func:`sklearn.pipeline.make_pipeline`.

.. code-block:: python

   >>> from sklearn.pipeline import make_pipeline
   >>> from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
   >>> from sklearn.linear_model import SGDClassifier

   >>> pipeline = make_pipeline(CountVectorizer(),
                                TfidfTransformer(),
                                SGDClassifier())
   >>> pipeline
   Pipeline(steps=[('countvectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False))])

Pipelines work by calling the usual ``fit`` and ``transform`` methods in succession.
The result of the prior ``transform`` is passed into the next ``fit`` step.
We'll see an example in the next section.

The common combination of pipelines and hyper-parameter search provide an
opportunity for dask to speed up model training.

Hyper-parameter Search
----------------------

Most scikit-learn estimators have a set of *hyper-parameters*.
These are parameters that are not learned during estimation; they must
be set ahead of time. :class:`sklearn.model_selection.GridSearchCV` and
:class:`sklearn.model_selection.RandomizedSearchCV` let you tune your
hyper-parameters by searching over the space of hyper-parameters to find the
combination that gives the best performance on a cross-validation set.

Here's where dask comes in: If you use the drop-in replacements
:class:`dask_searchcv.GridSearchCV` and
:class:`dask_searchcv.RandomizedSearchCV` to fit a ``Pipeline``, you can improve
the training time since dask will cache and reuse the intermediate steps.

.. code-block:: python

   >>> from dask_searchcv import GridSearchCV
   >>> param_grid = {
   ...     'tfidftransformer__norm': ['l1', 'l2', None],
   ...     'sgdclassifier__loss': ['hing', 'log'],
   ...     'sgdclassifier__alpha': [1e-5, 1e-3, 1e-1],
   ... }

   >>> clf = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
   GridSearchCV(cache_cv=True, cv=None, error_score='raise',
     estimator=Pipeline(steps=[('countvectorizer', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
     dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
     lowercase=True, max_df=1.0, max_features=None, min_df=1,
     ngram_range=(1, 1), preprocessor=None, stop_words=None,
     power_t=0.5, random_state=None, shuffle=True,
     verbose=0, warm_start=False))]),
     iid=True, n_jobs=-1,
     param_grid={'tfidftransformer__norm': ['l1', 'l2', None], 'sgdclassifier__loss': ['hing', 'log'], 'sgdclassifier__alpha': [1e-05, 0.001, 0.1]},
     refit=True, return_train_score=True, scheduler=None, scoring=None)

With the regular scikit-learn version, each stage of the pipeline must be fit
for each of the combinations of the parameters, even if that step isn't being
searched over. For example, the ``CountVectorizer`` must be fit 3 * 2 * 2 = 12
times, even though it's identical each time.

See :ref:`examples/hyperparameter-search.ipynb` for an example.

Incremental Learnings
---------------------

Some scikit-learn models support `incremental learning`_, they can see batches
of the datasets and update the parameters as new data comes in. This fits nicely
with dask's block-wise nature: dask arrays are composed of many smaller NumPy
arrays. ``dask-ml`` wraps scikit-learn's incremental learners, so that the usual
``.fit`` API will work on larger-than-memory datasets. These wrappers can be
dropped into a :class:`sklearn.pipeline.Pipeline` just like normal. In
``dask-ml``, all of these estimators are prefixed with ``Partial``, e.g.
:class:`PartialSGDClassifier`.

.. note::

   While these wrappers are useful for fitting on larger than memory datasets
   out-of-core, they *do not* support any kind of parallelism or distributed
   learning. Inside, e.g. ``PartialSGDClassifier.fit()``, execution is entirely
   sequential.

.. _dask-searchcv: http://dask-searchcv.readthedocs.io/en/latest/
.. _incremental learning: http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning
