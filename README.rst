dask-learn
==========

|Travis Status|

Tools for working with Scikit-Learn and Dask.

This library provides implementations of Scikit-Learn's ``GridSearchCV`` and
``RandomizedGridSearchCV``. They implement many (but not all) of the same
parameters, and should be a drop-in replacement for the subset that they do
implement. For certain problems, these implementations can be more efficient
than those in scikit-learn, as they can avoid repeating expensive repeated
comptuations.

.. |Travis Status| image:: https://travis-ci.org/dask/dask-learn.svg?branch=master
   :target: https://travis-ci.org/dask/dask-learn
