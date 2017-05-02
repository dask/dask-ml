.. currentmodule:: dask_glm

.. _api-reference:

API Reference
-------------

.. _api.estimators:

Estimators
==========

.. automodule:: dask_glm.estimators
   :members:

.. _api.families:

Families
========

.. automodule:: dask_glm.families
   :members:

.. _api.algorithms:

Algorithms
==========

.. automodule:: dask_glm.algorithms
   :members:

.. _api.regularizers:

Regularizers
============

.. _api.regularizers.available:

Available ``Regularizers``
~~~~~~~~~~~~~~~~~~~~~~~~~~

These regularizers are included with dask-glm.

.. automodule:: dask_glm.regularizers
   :members:
   :exclude-members: Regularizer

.. _api.regularizers.interface:

``Regularizer`` Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

Users wishing to implement their own regularizer should
satisfy this interface.

.. autoclass:: dask_glm.regularizers.Regularizer
   :members:

