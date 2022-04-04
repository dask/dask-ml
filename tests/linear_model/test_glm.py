import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model as slm
from dask.dataframe.utils import assert_eq
from dask_glm.regularizers import Regularizer
from sklearn.pipeline import make_pipeline

from dask_ml.datasets import make_classification, make_counts, make_regression
from dask_ml.linear_model import LinearRegression, LogisticRegression, PoissonRegression
from dask_ml.linear_model.utils import add_intercept
from dask_ml.model_selection import GridSearchCV


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def solver(request):
    """Parametrized fixture for all the solver names"""
    return request.param


@pytest.fixture(params=[r() for r in Regularizer.__subclasses__()])
def regularizer(request):
    """Parametrized fixture for all the regularizer names"""
    return request.param


class DoNothingTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        return X

    def get_params(self, deep=True):
        return {}


X, y = make_classification(chunks=50)


@pytest.fixture()
def glm_regression_data():
    X, y = make_regression(n_samples=1000)


def test_lr_init(solver):
    LogisticRegression(solver=solver)


def test_pr_init(solver):
    PoissonRegression(solver=solver)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit(fit_intercept, solver):
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    lr.predict_proba(X)


@pytest.mark.parametrize(
    "solver", ["admm", "newton", "lbfgs", "proximal_grad", "gradient_descent"]
)
def test_fit_solver(solver):
    import dask_glm
    import packaging.version

    if packaging.version.parse(dask_glm.__version__) <= packaging.version.parse(
        "0.2.0"
    ):
        pytest.skip("FutureWarning for dask config.")

    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression(solver=solver)
    lr.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lm(fit_intercept):
    X, y = make_regression(n_samples=100, n_features=5, chunks=50)
    lr = LinearRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.predict(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_big(fit_intercept):
    X, y = make_classification(chunks=50)
    lr = LogisticRegression(fit_intercept=fit_intercept)
    lr.fit(X, y)
    lr.decision_function(X)
    lr.predict(X)
    lr.predict_proba(X)
    if fit_intercept:
        assert lr.intercept_ is not None


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_poisson_fit(fit_intercept):
    X, y = make_counts(n_samples=100, chunks=500)
    pr = PoissonRegression(fit_intercept=fit_intercept)
    pr.fit(X, y)
    pr.predict(X)
    pr.get_deviance(X, y)
    if fit_intercept:
        assert pr.intercept_ is not None


def test_in_pipeline():
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    pipe.fit(X, y)


def test_gridsearch():
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    grid = {"logisticregression__C": [1000, 100, 10, 2]}
    pipe = make_pipeline(DoNothingTransformer(), LogisticRegression())
    search = GridSearchCV(pipe, grid, cv=3)
    search.fit(X, y)


def test_add_intercept_dask_dataframe():
    X = dd.from_pandas(pd.DataFrame({"A": [1, 2, 3]}), npartitions=2)
    result = add_intercept(X)
    expected = dd.from_pandas(
        pd.DataFrame(
            {"intercept": [1, 1, 1], "A": [1, 2, 3]}, columns=["intercept", "A"]
        ),
        npartitions=2,
    )
    assert_eq(result, expected)

    df = dd.from_pandas(pd.DataFrame({"intercept": [1, 2, 3]}), npartitions=2)
    with pytest.raises(ValueError):
        add_intercept(df)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_unknown_chunks_ok(fit_intercept):
    # https://github.com/dask/dask-ml/issues/145
    X = dd.from_pandas(pd.DataFrame(np.random.uniform(size=(10, 5))), 2).values
    y = dd.from_pandas(pd.Series(np.random.uniform(size=(10,))), 2).values

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y)


def test_add_intercept_unknown_ndim():
    X = dd.from_pandas(pd.DataFrame(np.ones((10, 5))), 2).values
    result = add_intercept(X)
    expected = np.ones((10, 6))
    da.utils.assert_eq(result, expected)


def test_add_intercept_raises_ndim():
    X = da.random.uniform(size=10, chunks=5)

    with pytest.raises(ValueError) as m:
        add_intercept(X)

    assert m.match("'X' should have 2 dimensions")


def test_add_intercept_raises_chunks():
    X = da.random.uniform(size=(10, 4), chunks=(4, 2))

    with pytest.raises(ValueError) as m:
        add_intercept(X)

    assert m.match("Chunking is only allowed")


def test_add_intercept_ordering():
    """Tests that add_intercept gives same result for dask / numpy objects"""
    X_np = np.arange(100).reshape(20, 5)
    X_da = da.from_array(X_np, chunks=(20, 5))
    np_result = add_intercept(X_np)
    da_result = add_intercept(X_da)
    da.utils.assert_eq(np_result, da_result)


def test_lr_score():
    X = da.from_array(np.arange(1000).reshape(1000, 1))
    lr = LinearRegression()
    lr.fit(X, X)
    assert lr.score(X, X) == pytest.approx(1, 0.001)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_dataframe_warns_about_chunks(fit_intercept):
    rng = np.random.RandomState(42)
    n, d = 20, 5
    kwargs = dict(npartitions=4)
    X = dd.from_pandas(pd.DataFrame(rng.uniform(size=(n, d))), **kwargs)
    y = dd.from_pandas(pd.Series(rng.choice(2, size=n)), **kwargs)
    clf = LogisticRegression(fit_intercept=fit_intercept)
    msg = "does not support dask dataframes.*might be resolved with"
    with pytest.raises(TypeError, match=msg):
        clf.fit(X, y)
    clf.fit(X.values, y.values)
    clf.fit(X.to_dask_array(), y.to_dask_array())
    clf.fit(X.to_dask_array(lengths=True), y.to_dask_array(lengths=True))


def test_logistic_predict_proba_shape():
    X, y = make_classification(n_samples=100, n_features=5, chunks=50)
    lr = LogisticRegression()
    lr.fit(X, y)
    prob = lr.predict_proba(X)
    assert prob.shape == (100, 2)


@pytest.mark.parametrize(
    "reg_type,data_generator",
    [
        (LinearRegression, "single_chunk_regression"),
        (LogisticRegression, "single_chunk_classification"),
        (PoissonRegression, "single_chunk_count_classification"),
    ],
)
def test_model_coef_dask_numpy(reg_type, data_generator, request):
    """Tests that models return same coefficients and intercepts with array types"""
    X, y = request.getfixturevalue(data_generator)
    np_mod, da_mod = reg_type(fit_intercept=True), reg_type(fit_intercept=True)
    da_mod.fit(X, y)
    np_mod.fit(X.compute(), y.compute())
    np.testing.assert_allclose(da_mod.coef_, np_mod.coef_)
    np.testing.assert_allclose(da_mod.intercept_, np_mod.intercept_)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_type, skl_reg_type, skl_params, data_generator",
    [
        (LinearRegression, slm.LinearRegression, {}, "single_chunk_regression"),
        (
            LogisticRegression,
            slm.LogisticRegression,
            {"penalty": "none"},
            "single_chunk_classification",
        ),
        (
            PoissonRegression,
            slm.PoissonRegressor,
            {"alpha": 0},
            "single_chunk_count_classification",
        ),
    ],
)
def test_model_against_sklearn(
    reg_type, skl_reg_type, skl_params, data_generator, fit_intercept, request
):
    """
    Tests accuracy of model predictions

    Notes:
        1. Dask ML 'newton' solver does not implement a penalty, only
            set arbitrarily high for clarity of non-regularization.
    """
    X, y = request.getfixturevalue(data_generator)
    dask_ml_model = reg_type(
        fit_intercept=fit_intercept, solver="newton", penalty="l2", C=1e8, tol=1e-8
    )
    skl_model = skl_reg_type(fit_intercept=fit_intercept, **skl_params)
    # skl_model has to be fit with numpy data
    dask_ml_model.fit(X, y)
    skl_model.fit(X.compute(), y.compute())
    np.testing.assert_allclose(
        dask_ml_model.predict(X), skl_model.predict(X), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize(
    "reg_type, skl_reg_type, skl_params, dataset_function",
    [
        (LinearRegression, slm.LinearRegression, {}, make_regression),
        (
            LogisticRegression,
            slm.LogisticRegression,
            {"penalty": "none"},
            make_classification,
        ),
    ],
)
def test_model_coef_against_sklearn(
    reg_type, skl_reg_type, skl_params, dataset_function
):
    """
    Tests accuracy of coefficient calculation.

    sk-learn least squares method does not tie out to iterative solvers with
    e.g. Rdi
    """
    X, y = dataset_function(
        n_samples=100, n_features=20, n_informative=10, chunks=100, random_state=0
    )
    X_np, y_np = X.compute(), y.compute()

    skl_model = skl_reg_type(fit_intercept=True, **skl_params)
    skl_model.fit(X_np, y_np)

    dask_ml_model = reg_type(fit_intercept=True, solver="newton", penalty="l2", C=1e8)
    dask_ml_model.fit(X, y)

    np.testing.assert_allclose(
        skl_model.coef_.flatten(), dask_ml_model.coef_, rtol=1e-3, atol=1e-8
    )


def test_poisson_regressor_against_sklearn(single_chunk_count_classification):
    X, y = single_chunk_count_classification
    skl_model = slm.PoissonRegressor(alpha=0, fit_intercept=True)
    skl_model.fit(X, y)
    dask_ml_model = PoissonRegression(
        fit_intercept=True, solver="newton", penalty="l2", C=1e8
    )
    dask_ml_model.fit(X, y)
    np.testing.assert_allclose(
        skl_model.coef_.flatten(), dask_ml_model.coef_, rtol=1e-3, atol=1e-3
    )
