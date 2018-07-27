.. _roadmap:

Dask-ML Roadmap
===============

Dask-ML wants to enable scalable machine learning in Python. It aims to do so by

1. Working with existing libraries within the Python ecosystem
2. Using the features of Dask to scale computation to larger datasets and larger
   problems

We'll use these two values to guide what's appropriate for inclusion in Dask-ML.

Scalable Algorithms
-------------------

Some algorithms are especially (or only) useful with large datasets, and so may
not be implemented in scikit-learn, e.g. [High Cardinality Encoders](https://github.com/dask/dask-ml/issues/120).

Others scalable algorithms might be unique to Dask Array and Dask DataFrame's
blocked structure (e.g. https://github.com/dask/dask-ml/issues/135).

Text Processing
---------------

We'd like to develop examples of working with large text datasets. Developing
these examples might motivate some additional estimators and utilities that
should be included in Dask-ML.

Integrations with Other Frameworks
----------------------------------

Other Deep Learning and Machine Learning frameworks have their own distributed
runtimes. Examples include Tensorflow, PyTorch, XGBoost, and LightGBM. Dask-ML
is not interested in re-implementing everything these libraries too. Rather,
we'd like to provide integrations so that Dask's collections can be seamlessly
transferred from a Dask cluster to one of those framework's distributed runtime.
This should be as seamless as using, say, a NumPy array on a single machine.

Asynchronous Training
---------------------

Dask's distributed scheduler allows for asynchonously submitting tasks, and
later awaiting the results. We explore user-APIs and optimization algorithms
that can take advantage of this.
