from gbrt.utils import make_data
from gbrt.ensemble import GradientBoostingRegressor

X, y = make_data()


def test_smoke():
    est = GradientBoostingRegressor()
    est.fit(X, y)
