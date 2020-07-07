import numpy as np
import numpy.testing as npt
import pytest

from dask_ml.linear_model import regularizers as regs


@pytest.mark.parametrize(
    "func,args",
    [("f", [0]), ("gradient", [0]), ("hessian", [0]), ("proximal_operator", [0, 1])],
)
def test_base_class_raises_notimplementederror(func, args):
    with pytest.raises(NotImplementedError):
        getattr(regs.Regularizer(), func)(*args)


class FooRegularizer(regs.Regularizer):
    def f(self, beta):
        return beta + 1

    def gradient(self, beta):
        return beta + 1

    def hessian(self, beta):
        return beta + 1


@pytest.mark.parametrize("func", ["add_reg_f", "add_reg_grad", "add_reg_hessian"])
def test_add_reg_funcs(func):
    def foo(x):
        return x ** 2

    new_func = getattr(FooRegularizer(), func)(foo, 1)
    assert callable(new_func)
    assert new_func(2) == 7


def test_regularizer_get_passes_through_instance():
    x = FooRegularizer()
    assert regs.Regularizer.get(x) == x


def test_regularizer_get_unnamed_raises():
    with pytest.raises(KeyError):
        regs.Regularizer.get("foo")


def test_regularizer_gets_from_name():
    class Foo(regs.Regularizer):
        name = "foo"

    assert isinstance(regs.Regularizer.get("foo"), Foo)


@pytest.mark.parametrize(
    "beta,expected", [(np.array([0, 0, 0]), 0), (np.array([1, 2, 3]), 7)]
)
def test_l2_function(beta, expected):
    assert regs.L2().f(beta) == expected


@pytest.mark.parametrize("beta", [np.array([0, 0, 0]), np.array([1, 2, 3])])
def test_l2_gradient(beta):
    npt.assert_array_equal(regs.L2().gradient(beta), beta)


@pytest.mark.parametrize("beta", [np.array([0, 0, 0]), np.array([1, 2, 3])])
def test_l2_hessian(beta):
    npt.assert_array_equal(regs.L2().hessian(beta), np.eye(len(beta)))


@pytest.mark.parametrize(
    "beta,expected",
    [
        (np.array([0, 0, 0]), np.array([0, 0, 0])),
        (np.array([1, 2, 3]), np.array([0.5, 1, 1.5])),
    ],
)
def test_l2_proximal_operator(beta, expected):
    npt.assert_array_equal(regs.L2().proximal_operator(beta, 1), expected)


@pytest.mark.parametrize(
    "beta,expected", [(np.array([0, 0, 0]), 0), (np.array([-1, 2, 3]), 6)]
)
def test_l1_function(beta, expected):
    assert regs.L1().f(beta) == expected


@pytest.mark.parametrize(
    "beta,expected",
    [
        (np.array([1, 2, 3]), np.array([1, 1, 1])),
        (np.array([-1, 2, 3]), np.array([-1, 1, 1])),
    ],
)
def test_l1_gradient(beta, expected):
    npt.assert_array_equal(regs.L1().gradient(beta), expected)


@pytest.mark.parametrize(
    "beta",
    [np.array([0.00000001, 1, 2]), np.array([-0.00000001, 1, 2]), np.array([0, 0, 0])],
)
def test_l1_gradient_raises_near_zero(beta):
    with pytest.raises(ValueError):
        regs.L1().gradient(beta)


def test_l1_hessian():
    npt.assert_array_equal(
        regs.L1().hessian(np.array([1, 2])), np.array([[0, 0], [0, 0]])
    )


def test_l1_hessian_raises():
    with pytest.raises(ValueError):
        regs.L1().hessian(np.array([0, 0, 0]))


@pytest.mark.parametrize(
    "beta,expected",
    [
        (np.array([0, 0, 0]), np.array([0, 0, 0])),
        (np.array([1, 2, 3]), np.array([0, 1, 2])),
    ],
)
def test_l1_proximal_operator(beta, expected):
    npt.assert_array_equal(regs.L1().proximal_operator(beta, 1), expected)


@pytest.mark.parametrize(
    "beta,expected", [(np.array([0, 0, 0]), 0), (np.array([1, 2, 3]), 6.5)]
)
def test_elastic_net_function(beta, expected):
    assert regs.ElasticNet().f(beta) == expected


def test_elastic_net_function_zero_weight_is_l2():
    beta = np.array([1, 2, 3])
    assert regs.ElasticNet(weight=0).f(beta) == regs.L2().f(beta)


def test_elastic_net_function_zero_weight_is_l1():
    beta = np.array([1, 2, 3])
    assert regs.ElasticNet(weight=1).f(beta) == regs.L1().f(beta)


def test_elastic_net_gradient():
    beta = np.array([1, 2, 3])
    npt.assert_array_equal(
        regs.ElasticNet(weight=0.5).gradient(beta), np.array([1, 1.5, 2])
    )


def test_elastic_net_gradient_zero_weight_is_l2():
    beta = np.array([1, 2, 3])
    npt.assert_array_equal(
        regs.ElasticNet(weight=0).gradient(beta), regs.L2().gradient(beta)
    )


def test_elastic_net_gradient_zero_weight_is_l1():
    beta = np.array([1, 2, 3])
    npt.assert_array_equal(
        regs.ElasticNet(weight=1).gradient(beta), regs.L1().gradient(beta)
    )


def test_elastic_net_hessian():
    beta = np.array([1, 2, 3])
    npt.assert_array_equal(
        regs.ElasticNet(weight=0.5).hessian(beta),
        np.eye(len(beta)) * regs.ElasticNet().weight,
    )


def test_elastic_net_hessian_raises():
    with pytest.raises(ValueError):
        regs.ElasticNet(weight=0.5).hessian(np.array([0, 1, 2]))


def test_elastic_net_proximal_operator():
    beta = np.array([1, 2, 3])
    npt.assert_array_equal(regs.ElasticNet(weight=0.5).proximal_operator(beta, 1), beta)
