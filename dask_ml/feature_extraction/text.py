"""
Utilities to build feature vectors from text documents.
"""
import itertools

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import scipy.sparse
import sklearn.base
import sklearn.feature_extraction.text
from sklearn.utils.validation import check_is_fitted
from distributed import get_client


class _BaseHasher(sklearn.base.BaseEstimator):
    @property
    def _hasher(self) -> sklearn.base.BaseEstimator:
        """
        Abstract method for subclasses.

        Returns
        -------
        BaseEstimator
            Should be the *class* (not an instance) to be used for
            transforming blocks of the input.
        """
        raise NotImplementedError

    def _transformer(self, part):
        return self._hasher(**self.get_params()).transform(part)

    def transform(self, raw_X):
        msg = "'X' should be a 1-dimensional array with length 'num_samples'."

        if not dask.is_dask_collection(raw_X):
            return self._hasher(**self.get_params()).transform(raw_X)

        if isinstance(raw_X, db.Bag):
            bag2 = raw_X.map_partitions(self._transformer)
            objs = bag2.to_delayed()
            arrs = [
                da.from_delayed(obj, (np.nan, self.n_features), self.dtype)
                for obj in objs
            ]
            result = da.concatenate(arrs, axis=0)
        elif isinstance(raw_X, dd.Series):
            result = raw_X.map_partitions(self._transformer)
        elif isinstance(raw_X, da.Array):
            # dask.Array
            chunks = ((np.nan,) * raw_X.numblocks[0], (self.n_features,))
            if raw_X.ndim == 1:
                result = raw_X.map_blocks(
                    self._transformer, dtype="f8", chunks=chunks, new_axis=1
                )
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

        meta = scipy.sparse.eye(0, format="csr")
        result._meta = meta
        return result



class HashingVectorizer(_BaseHasher, sklearn.feature_extraction.text.HashingVectorizer):
    # explicit doc for Sphinx
    __doc__ = sklearn.feature_extraction.text.HashingVectorizer.__doc__

    @property
    def _hasher(self):
        return sklearn.feature_extraction.text.HashingVectorizer

    def transform(self, raw_X):
        """Transform a sequence of documents to a document-term matrix.

        Transformation is done in parallel, and correctly handles dask
        collections.

        Parameters
        ----------
        raw_X : dask.bag.Bag or dask.dataframe.Series, length = n_samples
            Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        Returns
        -------
        X : dask.array.Array, shape = (n_samples, self.n_features)
            Document-term matrix. Each block of the array is a scipy sparse
            matrix.

        Notes
        -----
        The returned dask Array is composed scipy sparse matricies. If you need
        to compute on the result immediately, you may need to convert the individual
        blocks to ndarrays or pydata/sparse matricies.

        >>> import sparse
        >>> X.map_blocks(sparse.COO.from_scipy_sparse, dtype=X.dtype)  # doctest: +SKIP

        See the :doc:`examples/text-vectorization` for more.
        """
        return super().transform(raw_X)


class FeatureHasher(_BaseHasher, sklearn.feature_extraction.text.FeatureHasher):
    __doc__ = sklearn.feature_extraction.text.FeatureHasher.__doc__

    @property
    def _hasher(self):
        return sklearn.feature_extraction.text.FeatureHasher


class Vocabulary:
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            self.vocabulary = set()
            self.fixed_vocabulary = True
        else:
            self.vocabulary = vocabulary

    def update(self, obj):
        assert not self.fixed_vocabulary
        self.vocabulary |= obj

    def finalize(self):
        if self.fixed_vocabulary:
            return self.vocabulary
        else:
            return {key: i for i, key in enumerate(sorted(self.vocabulary))}


class CountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    # Potential issues:
    # 1. The vocabularies might be somewhat large.
    #    Consider storing this on the cluster, rather than shipping between
    #    the client?
    def fit_transform(self, raw_documents, y=None):
        # Just bag for now.
        # Two cases:
        # 1. vocabulary provided, easy.
        # 2. vocabulary learned, harder.
        params = self.get_params()
        vocabulary = params.pop("vocabulary")

        client = get_client()

        if self.vocabulary is not None:
            # Case 1: Just map transform.
            fixed_vocabulary = True
            vocabulary_ = client.submit(Vocabulary, vocabulary=vocabulary, actor=True)
            vocabulary_actor = vocabulary_.result()

            result = raw_documents.map_partitions(_count_vectorizer_transform,
                                                  vocabulary_actor, params)
        else:
            fixed_vocabulary = False
            # Case 2: learn vocabulary from the data.
            vocabularies = raw_documents.map_partitions(_build_vocabulary, vocabulary, params)
            # compute and merge.
            dask.compute(vocabularies)  # List[Set[str]]
            vocabulary = 
            # maybe the merge should be done as a task in the cluster.
            vocabulary = {key: i for i, key in enumerate(set(itertools.chain.from_iterable(vocabularies)))}
            # same as before
            result = raw_documents.map_partitions(_count_vectorizer_transform,
                                                  vocabulary, params)

        meta = scipy.sparse.eye(0, format="csr", dtype=self.dtype)
        result = build_array(result, len(vocabulary), meta)

        # XXX: more params
        self.vocabulary_ = vocabulary
        self.fixed_vocabulary_ = fixed_vocabulary

        return result

    def transform(self, raw_documents):
        params = self.get_params()
        vocabulary = params.pop("vocabulary")
        if vocabulary is None:
            check_is_fitted(self, "vocabulary_")
            vocabulary = self.vocabulary_
        result = raw_documents.map_partitions(_count_vectorizer_transform, vocabulary, params)
        meta = scipy.sparse.eye(0, format="csr", dtype=self.dtype)
        return build_array(result, len(vocabulary), meta)


def _count_vectorizer_transform(partition, vocabulary_actor, params):
    vocabulary = vocabulary_actor.finalize()
    model = sklearn.feature_extraction.text.CountVectorizer(vocabulary=vocabulary,
                                                            **params)
    return model.transform(partition)


def _build_vocabulary(partition, vocabulary_actor, params):
    model = sklearn.feature_extraction.text.CountVectorizer(**params)
    model.fit(partition)
    vocabulary_actor.update(set(model.vocabulary_))


def build_array(b, n_features, meta):
    """
    Build a Dask Array from a bag of scipy.sparse matrics.
    """
    objs = b.to_delayed()
    arrs = [da.from_delayed(obj, (np.nan, n_features), meta=meta)
            for obj in objs]
    arr = da.concatenate(arrs)
    return arr
