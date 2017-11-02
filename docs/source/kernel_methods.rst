Kernel Methods
==============

.. currentmodule:: dask_ml.kernel_approximation

.. autosummary::

   Nystroem

Kernel methods are at an important component of many machine learning
algorithms, including Support Vector Machines, Gaussian Processes, and Spectral
Clustering.

Unfortunately, naïve implementations of these algorithms don't scale to larger
datasets. Given a dataset that has the shape ``(n_samples, n_features)``, an
*exact* kernel-based solution has ``O(n_samples ** 3)`` complexity. The kernel
matrix alone is ``(n_samples, n_samples)``, which can be impractical to
calculate or store for datasets with many samples.

We can sometimes use *approximate* kernel methods; we trade some accuracy for
much improved training and prediction times [1]_.

.. ipython:: python

   from dask_ml.kernel_approximation import Nystroem
   import dask.array as da

   X = da.random.uniform(size=(10000, 50), chunks=(1000, 50))
   est = Nystroem()
   est.fit(X)
   est

All of the attributes learned on :class:`Nystroem` are small NumPy arrays. We've
sampled (uniformly at random) ``n_components`` rows from the large dask array
``X``, and used those for the approximation. In general, increasing
``n_components`` will improve the approximation, but will take longer for
training and transforming.


.. topic:: References

   .. [1] Williams and Seeger (2000). Using the Nyström Method to Speed Up
          Kernel Machines. Advances in Neural Information Processing Systems 13
          (NIPS 2000)
          http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.18.7519
