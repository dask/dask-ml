# dask-learn

[![Build Status](https://travis-ci.org/dask/dask-learn.svg?branch=master)](https://travis-ci.org/dask/dask-learn)

Tools for working with scikit-learn and dask.

This library came out of some proof-of-concept experimentation. None of it
should be considered stable, but certain parts are more solid than others.

### Parts that can be considered ready to use:

- `dklearn.Pipeline`: a drop in replacement for `sklearn.Pipeline`. Multiple
  pipelines can be merged together, meaning that fitting the same estimator
  multiple times can be avoided.

- `dklearn.grid_search.GridSearchCV` and
  `dklearn.grid_search.RandomizedSearchCV` mirror their scikit-learn
  counterparts. They implement many (but not all) of the same parameters, and
  should be a drop-in replacement for the subset that they do implement. For
  certain problems, these implementations can be more efficient than those in
  scikit-learn, as they can avoid repeating expensive repeated comptuations.
  For more information, see [this blog
  post](http://jcrist.github.io/dask-sklearn-part-1.html).

- `dklearn.feature_extraction.FeatureHasher` and
  `dklearn.feature_extraction.HashingVectorizer` mirror their scikit-learn
  counterparts, and should work fine for the common dask collections.

### Parts that haven't seen as much use, and could use some love:

- `dklearn.Averaged` and `dklearn.Chained`. `Averaged` wraps a scikit-learn
  estimator, and fits it in parallel across training data, averaging the
  coefficients at the end. `Chained` wraps any scikit-learn estimator that
  implements the [incremental
  learning](http://scikit-learn.org/stable/modules/scaling_strategies.html#incremental-learning)
  api (i.e. the `partial_fit` method). The estimator is then incrementally fit
  on each chunk of the training data. For more information on these
  components, see [this blog
  post](http://jcrist.github.io/dask-sklearn-part-2.html).

- `dklearn.cross_validation` implements both `KFold` and `RandomSplit` for the
  common dask collections. `train_test_split` is also implemented.
