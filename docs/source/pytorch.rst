.. _pytorch:

PyTorch
=======

Skorch_ brings a Scikit-learn API to PyTorch_. Skorch allows PyTorch models to
be wrapped in Scikit-learn compatible estimators. So, that means that PyTorch
models wrapped in Skorch can be used with the rest of the Dask-ML API.  For
example, using Dask-ML's :class:`~dask_ml.model_selection.HyperbandSearchCV` or
:class:`~dask_ml.model_selection.Incremental` with PyTorch is possible after
wrapping with Skorch.

We encourage looking at the Skorch documentation for complete details.

Example usage
-------------

First, let's create a normal PyTorch model:

.. code-block:: python


   import torch.nn as nn
   import torch.nn.functional as F

   class ShallowNet(nn.Module):
       def __init__(self, n_features=5):
           super().__init__()
           self.layer1 = nn.Linear(n_features, 1)

       def forward(self, x):
           return F.relu(self.layer1(x))

With this, it's easy to use Skorch:

.. code-block:: python

   from skorch import NeuralNetRegressor
   import torch.optim as optim

   niceties = {
       "callbacks": False,
       "warm_start": False,
       "train_split": None,
       "max_epochs": 1,
   }

   model = NeuralNetRegressor(
       module=ShallowNet,
       module__n_features=5,
       criterion=nn.MSELoss,
       optimizer=optim.SGD,
       optimizer__lr=0.1,
       optimizer__momentum=0.9,
       batch_size=64,
       **niceties,
   )

Each parameter that the PyTorch ``nn.Module`` takes is prefixed with ``module__``,
and same for the optimizer (``optim.SGD`` takes a ``lr`` and ``momentum``
parameters). The ``niceties`` make sure Skorch uses all the data for training
and doesn't print excessive amounts of logs.

Now, this model can be used with Dask-ML. For example, it's possible to do the
following:

* Use PyTorch with the Dask-ML's model selection, including
  :class:`~dask_ml.model_selection.HyperbandSearchCV`.
* Use PyTorch with Dask-ML's :class:`~dask_ml.wrappers.Incremental`.

.. _Skorch: https://skorch.readthedocs.io/en/stable/
.. _PyTorch: https://pytorch.org
