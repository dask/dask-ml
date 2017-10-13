Hyper Parameter Search
======================

.. autosummary::
   sklearn.pipeline.make_pipeline
   sklearn.model_selection.GridSearchCV
   dask_ml.model_selection.GridSearchCV
   sklearn.model_selection.RandomizedSearchCV
   dask_ml.model_selection.RandomizedSearchCV

Most estimators have a set of *hyper-parameters*.
These are parameters that are not learned during training but instead must be
set ahead of time. Traditionally we use Scikit-Learn tools like
:class:`sklearn.model_selection.GridSearchCV` and
:class:`sklearn.model_selection.RandomizedSearchCV` to tune our
hyper-parameters by searching over the space of hyper-parameters to find the
combination that gives the best performance on a cross-validation set.

Pipelines
---------

This search for hyper-parameters can become significantly more expensive when
we have not a single estimator, but many estimators arranged into a pipeline.
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

Efficient Search
----------------

However now each of our estimators in our pipeline have hyper-parameters,
both expanding the space over which we want to search as well as adding
hierarchy to the search process.  For every parameter we try in the first stage
in the pipeline we want to try several in the second, and several more in the
third, and so on.

The common combination of pipelines and hyper-parameter search provide an
opportunity for dask to speed up model training not just by simple parallelism,
but also by searching the space in a more structured way.

If you use the drop-in replacements
:class:`dask_ml.model_selection.GridSearchCV` and
:class:`dask_ml.model_selection.RandomizedSearchCV` to fit a ``Pipeline``, you can improve
the training time since Dask will cache and reuse the intermediate steps.

.. code-block:: python

   >>> # from sklearn.model_selection import GridSearchCV  # replace import
   >>> from dask_ml.model_selection import GridSearchCV
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

.. _dask-searchcv: http://dask-searchcv.readthedocs.io/en/latest/
