import pytest
from distributed.utils_test import gen_cluster
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.datasets import make_regression

from dask_ml.model_selection import IncrementalSearchCV

try:
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from skorch import NeuralNetRegressor
except ImportError:
    pytestmark = pytest.mark.skip(reason="Missing pytorch or skorch.")

else:

    class ShallowNet(nn.Module):
        def __init__(self, n_features=5):
            super().__init__()
            self.layer1 = nn.Linear(n_features, 1)

        def forward(self, x):
            return F.relu(self.layer1(x))


@gen_cluster(client=True)
def test_pytorch(c, s, a, b):

    n_features = 10
    defaults = {
        "callbacks": False,
        "warm_start": False,
        "train_split": None,
        "max_epochs": 1,
    }
    model = NeuralNetRegressor(
        module=ShallowNet,
        module__n_features=n_features,
        criterion=nn.MSELoss,
        optimizer=optim.SGD,
        optimizer__lr=0.1,
        batch_size=64,
        **defaults,
    )

    model2 = clone(model)
    assert model.callbacks is False
    assert model.warm_start is False
    assert model.train_split is None
    assert model.max_epochs == 1

    params = {"optimizer__lr": loguniform(1e-3, 1e0)}
    X, y = make_regression(n_samples=100, n_features=n_features)
    X = X.astype("float32")
    y = y.astype("float32").reshape(-1, 1)
    search = IncrementalSearchCV(model2, params, max_iter=5, decay_rate=None)
    yield search.fit(X, y)
    assert search.best_score_ >= 0
