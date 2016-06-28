from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

import pytest
import numpy as np
import dask.bag as db
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn import feature_extraction
from numpy.testing import assert_array_equal

import dklearn.matrix as dm
from dklearn import from_sklearn
from dklearn.feature_extraction import FeatureHasher, HashingVectorizer

import dask
dask.set_options(get=dask.get)


class WrapperHarness(object):
    def test_from_sklearn(self):
        d = from_sklearn(self.sk)
        assert from_sklearn(self.sk)._name == d._name
        assert from_sklearn(self.sk2)._name != d._name

        with pytest.raises(TypeError):
            self.cls.from_sklearn("not a sklearn estimator")

    def test__init__(self):
        d = self.cls(n_features=16)
        assert d._name == from_sklearn(self.sk)._name
        assert d._name != self.cls(n_features=20)._name

    def test_tokenize(self):
        d = from_sklearn(self.sk)
        d2 = from_sklearn(self.sk2)
        assert tokenize(d) == tokenize(d)
        assert tokenize(d) != tokenize(d2)

    def test_clone(self):
        d = from_sklearn(self.sk)
        d2 = clone(d)
        assert d._name == d2._name
        assert d.get_params() == d2.get_params()
        assert d._est is not d2._est

    def test_get_params(self):
        d = from_sklearn(self.sk)
        assert d.get_params() == self.sk.get_params()
        assert d.get_params(deep=False) == self.sk.get_params(deep=False)

    def test_set_params(self):
        d = from_sklearn(self.sk)
        d2 = d.set_params(**self.sk2.get_params())
        assert isinstance(d2, type(d))
        # Check no mutation
        params = self.sk.get_params()
        assert d.get_params() == params
        assert d.compute().get_params() == params
        params2 = self.sk2.get_params()
        assert d2.get_params() == params2
        assert d2.compute().get_params() == params2

    def test_setattr(self):
        d = from_sklearn(self.sk)
        with pytest.raises(AttributeError):
            d.n_features = 20

    def test_getattr(self):
        d = from_sklearn(self.sk)
        assert d.n_features == self.sk.n_features
        with pytest.raises(AttributeError):
            d.not_a_real_parameter

    def test_dir(self):
        d = from_sklearn(self.sk)
        attrs = dir(d)
        assert 'n_features' in attrs

    def test_repr(self):
        d = from_sklearn(self.sk)
        assert repr(d) == repr(self.sk)

    def test_fit(self):
        d = from_sklearn(self.sk)
        b = db.from_sequence(self.raw_X)
        fit = d.fit(b, db.range(len(self.raw_X), len(self.raw_X)))
        assert fit is d

    def test_transform(self):
        d = from_sklearn(self.sk)

        # Single element in each partition
        b = db.from_sequence(self.raw_X)
        X1 = d.transform(b)
        assert X1.name == d.transform(b).name
        assert X1.name != d.transform(b.repartition(3)).name
        assert isinstance(X1, dm.Matrix)
        assert X1.shape == (None, 16)
        assert X1.dtype == np.dtype('f8')

        # Multiple element in each partition
        b2 = b.repartition(2)
        X2 = d.set_params(dtype='i8').transform(b2)
        assert X1.name != X2.name
        assert isinstance(X2, dm.Matrix)
        assert X2.shape == (None, 16)
        assert X2.dtype == np.dtype('i8')

        assert_array_equal(X1.compute().toarray(),
                           X2.compute().toarray().astype('f8'))

        # Delayed as input
        X3 = d.transform(self.raw_X)
        assert isinstance(X3, Delayed)
        assert X3._key == d.transform(self.raw_X)._key

        assert_array_equal(X1.compute().toarray(),
                           X3.compute().toarray())


class TestFeatureHasher(WrapperHarness):
    cls = FeatureHasher
    sk = feature_extraction.FeatureHasher(n_features=16)
    sk2 = feature_extraction.FeatureHasher(n_features=20)
    raw_X = [{"foo": 1, "dada": 2, "tzara": 3},
             {"foo": 4, "gaga": 5}] * 5


class TestHashingVectorizer(WrapperHarness):
    cls = HashingVectorizer
    sk = feature_extraction.text.HashingVectorizer(n_features=16)
    sk2 = feature_extraction.text.HashingVectorizer(n_features=20)
    raw_X = ["I don't like apples",
             "How do you like them apples",
             "I don't like green eggs and ham",
             "I do like apples",
             "Pears are fruit like apples"] * 2
