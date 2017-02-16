# dask-learn

[![Build Status](https://travis-ci.org/dask/dask-learn.svg?branch=master)](https://travis-ci.org/dask/dask-learn)

Tools for working with scikit-learn and dask.

This library provides implementations of Scikit-Learn's `GridSearchCV` and
`RandomizedGridSearchCV`. They implement many (but not all) of the same
parameters, and should be a drop-in replacement for the subset that they do
implement. For certain problems, these implementations can be more efficient
than those in scikit-learn, as they can avoid repeating expensive repeated
comptuations.  For more information, see [this blog
post](http://jcrist.github.io/dask-sklearn-part-1.html).
