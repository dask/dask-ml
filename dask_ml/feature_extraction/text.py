"""
Utilities to build feature vectors from text documents.
"""
import itertools

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import distributed
import numpy as np
import scipy.sparse
import sklearn.base
import sklearn.feature_extraction.text
from dask.delayed import Delayed
from distributed import Actor, get_client, wait
from sklearn.utils.validation import check_is_fitted


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
    vocabulary = set()
    fixed_vocabulary = None

    def __init__(self, vocabulary=None):
        if vocabulary is None:
            self.vocabulary = set()
            self.fixed_vocabulary = False
        else:
            self.fixed_vocabulary = True
            self.vocabulary = vocabulary

    def update(self, obj):
        assert not self.fixed_vocabulary
        self.vocabulary |= obj

    def finalize(self):
        if self.fixed_vocabulary:
            return self.vocabulary
        else:
            return {key: i for i, key in enumerate(sorted(self.vocabulary))}

    @property
    def n_features(self):
        return len(self.vocabulary)


class CountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    """Convert a collection of text documents to a matrix of token counts

    .. note::

       This implementation requires an active :class:`distributed.Client`.

    Notes
    -----
    When a vocabulary isn't provided, ``fit_transform`` requires two
    passes over the dataset: one to learn the vocabulary and a second
    to transform the data. Consider persisting the data if it fits
    in (distributed) memory.

    See Also
    --------
    sklearn.feature_extraction.text.CountVectorizer

    Examples
    --------
    >>> from dask_ml.feature_extraction.text import CountVectorizer
    >>> import dask.bag as db
    >>> from distributed import Client
    >>> client = Client()
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> corpus = db.from_sequence(corpus, npartitions=2)
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    dask.array<concatenate, shape=(nan, 9), dtype=int64, chunksize=(nan, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])
    >>> vectorizer.get_feature_names()
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
        use_actors=True,
    ):
        self.use_actors = use_actors
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

    def fit_transform(self, raw_documents, y=None):
        # try:
        #     client = get_client()
        #     use_actors = True
        # except ValueError:
        #     use_actors = False
        use_actors = self.use_actors
        if use_actors:
            client = get_client()

        params = self.get_params()
        vocabulary = params.pop("vocabulary")
        del params["use_actors"]

        if use_actors:
            vocabulary_actor = client.submit(
                Vocabulary, vocabulary=vocabulary, actor=True
            )
            vocabulary_actor = vocabulary_actor.result()
            vocabulary_for_transform = vocabulary_actor
        else:
            vocabulary_actor = None
            vocabulary_for_transform = vocabulary

        if self.vocabulary is not None:
            # Case 1: Just map transform.
            fixed_vocabulary = True
            n_features = vocabulary_length(vocabulary)
            vocabulary_ = vocabulary
        else:
            fixed_vocabulary = False
            # Case 2: learn vocabulary from the data.
            vocabularies = raw_documents.map_partitions(
                _build_vocabulary, vocabulary_actor, params
            )
            if use_actors:
                dask.compute(vocabularies)  # List[Set[str]]
                vocabulary_ = vocabulary_actor.finalize().result()
            else:
                vocabulary = vocabulary_for_transform = _merge_vocabulary(
                    *vocabularies.to_delayed()
                )
                vocabulary_ = vocabulary.compute()

        n_features = len(vocabulary_)
        result = raw_documents.map_partitions(
            _count_vectorizer_transform, vocabulary_for_transform, params
        )

        meta = scipy.sparse.eye(0, format="csr", dtype=self.dtype)
        result = build_array(result, n_features, meta)

        self.vocabulary_actor_ = vocabulary_actor
        self.vocabulary_ = vocabulary_
        self.fixed_vocabulary_ = fixed_vocabulary

        return result

    def transform(self, raw_documents):
        params = self.get_params()
        vocabulary = params.pop("vocabulary")
        use_actors = params.pop("use_actors")

        if vocabulary is None:
            check_is_fitted(self, "vocabulary_")
            if use_actors:
                vocabulary_for_transform = self.vocabulary_
            else:
                vocabulary_for_transform = self.vocabulary_actor_
        else:
            vocabulary_for_transform = vocabulary

        n_features = vocabulary_length(vocabulary)
        transformed = raw_documents.map_partitions(
            _count_vectorizer_transform, vocabulary_for_transform, params
        )
        meta = scipy.sparse.eye(0, format="csr", dtype=self.dtype)
        return build_array(transformed, n_features, meta)


def build_array(bag, n_features, meta):
    name = "from-bag-" + bag.name
    layer = {(name, i, 0): (k, i) for k, i in bag.__dask_keys__()}
    dsk = dask.highlevelgraph.HighLevelGraph.from_collections(
        name, layer, dependencies=[bag]
    )
    chunks = ((np.nan,) * bag.npartitions, (n_features,))
    return da.Array(dsk, name, chunks, meta=meta)


def vocabulary_length(vocabulary):
    if isinstance(vocabulary, dict):
        return len(vocabulary)
    elif isinstance(vocabulary, Vocabulary):
        return vocabulary.n_features
    elif isinstance(vocabulary, Delayed):
        try:
            return len(vocabulary)
        except ValueError:
            return len(vocabulary.compute())
    elif isinstance(vocabulary, distributed.Future):
        client = get_client()
        future = client.submit(len, vocabulary)
        wait(future)
        result = future.result()
        return result
    else:
        raise ValueError(f"Unknown vocabulary type {type(vocabulary)}.")


def _count_vectorizer_transform(partition, vocabulary, params):
    if isinstance(vocabulary, Actor):
        vocabulary = vocabulary.finalize().result()
    model = sklearn.feature_extraction.text.CountVectorizer(
        vocabulary=vocabulary, **params
    )
    return model.transform(partition)


def _build_vocabulary(partition, vocabulary_actor, params):
    model = sklearn.feature_extraction.text.CountVectorizer(**params)
    model.fit(partition)
    result = set(model.vocabulary_)
    if vocabulary_actor is not None:
        vocabulary_actor.update(result)
    else:
        return result


@dask.delayed
def _merge_vocabulary(*vocabularies):
    vocabulary = {
        key: i
        for i, key in enumerate(
            sorted(set(itertools.chain.from_iterable(vocabularies)))
        )
    }
    return vocabulary
