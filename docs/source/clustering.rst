.. _clustering:

Clustering
==========

.. currentmodule:: dask_ml.cluster

.. autosummary::
   KMeans
   PartialMiniBatchKMeans
   SpectralClustering

The :mod:`dask_ml.cluster` module implements several algorithms for clustering unlabeled data.

Spectral Clustering
-------------------

Spectral Clustering finds a low-dimensional embedding on the affinity matrix
between samples. The embedded dataset is then clustered, typically with KMeans.

Typically, spectral clustering algorithms do not scale well. Computing the
:math:`n_samples \times n_samples` affinity matrix becomes prohibitively
expensive when the number of samples is large. Several algorithms have been
proposed to work around this limitation.

In ``dask-ml``, we use the Nyström method to approximate the large affinity
matrix. This involves sampling ``n_components`` rows from the entire training
set. The exact affinity is computed for this subset
(:math:`n_components \times n_components`), and between this small subset and
the rest of the data (:math:`n_components \times (n_samples - n_components)`).
We avoid the direct computation of the rest of the affinity matrix.

Let :math:`S` be our :math:`n \times n` affinity matrix. We can rewrite that as

.. math::

  S_d = \left[
  \begin{array}\
      A   & B \\
      B^T & C \\
  \end{array}
  \right]

Where :math:`A` is the :math:`n \times n` affinity matrix of the
:math:`n_components` that we sampled, and :math:`B` is the
:math:`n \times (n - n_components)` affinity matrix between the sample and the
rest of the dataset. Instead of computing :math:`C` directly, we approximate it
with :math:`B^T A^{-1} B`.

See :doc:`auto_examples/plot_spectral_clustering` for an example showing how
:class:`dask_ml.cluster.SpectralClustering` scales in the number of samples.
