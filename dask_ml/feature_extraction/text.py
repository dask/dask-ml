"""
Utilities to build feature vectors from text documents.
"""
import itertools

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import distributed
import pandas as pd
import numpy as np
import scipy.sparse
import sklearn.base
import sklearn.feature_extraction.text
import sklearn.preprocessing
from dask.delayed import Delayed
from distributed import get_client, wait
from sklearn.utils.validation import check_is_fitted

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


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
        The returned dask Array is composed scipy sparse matrices. If you need
        to compute on the result immediately, you may need to convert the individual
        blocks to ndarrays or pydata/sparse matrices.

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


def _n_samples(X):
    """Count the number of samples in dask.array.Array X."""
    def chunk_n_samples(chunk, axis, keepdims):
        return np.array([chunk.shape[0]], dtype=np.int64)

    return da.reduction(X,
                        chunk=chunk_n_samples,
                        aggregate=np.sum,
                        concatenate=False,
                        dtype=np.int64)


def _document_frequency(X, dtype):
    """Count the number of non-zero values for each feature in dask array X."""
    def chunk_doc_freq(chunk, axis, keepdims):
        if scipy.sparse.isspmatrix_csr(chunk):
            return np.bincount(chunk.indices, minlength=chunk.shape[1])
        else:
            return np.diff(chunk.indptr)

    return da.reduction(X,
                        chunk=chunk_doc_freq,
                        aggregate=np.sum,
                        axis=0,
                        concatenate=False,
                        dtype=dtype)


class CountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    """Convert a collection of text documents to a matrix of token counts

    Notes
    -----
    When a vocabulary isn't provided, ``fit_transform`` requires two
    passes over the dataset: one to learn the vocabulary and a second
    to transform the data. Consider persisting the data if it fits
    in (distributed) memory prior to calling ``fit`` or ``transform``
    when not providing a ``vocabulary``.

    Additionally, this implementation benefits from having
    an active ``dask.distributed.Client``, even on a single machine.
    When a client is present, the learned ``vocabulary`` is persisted
    in distributed memory, which saves some recomputation and redundant
    communication.

    See Also
    --------
    sklearn.feature_extraction.text.CountVectorizer

    Examples
    --------
    The Dask-ML implementation currently requires that ``raw_documents``
    is either a :class:`dask.bag.Bag` of documents (lists of strings) or
    a :class:`dask.dataframe.Series` of documents (Series of strings)
    with partitions of type :class:`pandas.Series`.

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
    >>> corpus_bag = db.from_sequence(corpus, npartitions=2)
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus_bag)
    dask.array<concatenate, shape=(4, 9), dtype=int64, chunksize=(2, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])
    >>> vectorizer.get_feature_names()
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    
    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> corpus_dds = dd.from_pandas(pd.Series(corpus), npartitions=2)
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(corpus_dds)
    dask.array<concatenate, shape=(4, 9), dtype=int64, chunksize=(2, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])
    >>> vectorizer.get_feature_names()
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    """

    def get_CountVectorizer_params(self, deep=True):
        """
        Get CountVectorizer parameters (names and values) for this 
        estimator (self), whether it is an instance of CountVectorizer or an
        instance of a subclass of CountVectorizer.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the CountVectorizer parameters for this
            estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in CountVectorizer._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def fit_transform(self, raw_documents, y=None):
        params = self.get_CountVectorizer_params()
        vocabulary = params.pop("vocabulary")
        vocabulary_for_transform = vocabulary

        if self.vocabulary is not None:
            # Case 1: Just map transform.
            fixed_vocabulary = True
            n_features = vocabulary_length(vocabulary)
            vocabulary_ = vocabulary
        else:
            fixed_vocabulary = False
            # Case 2: learn vocabulary from the data.
            vocabularies = raw_documents.map_partitions(_build_vocabulary, params)
            vocabulary = vocabulary_for_transform = (
                _merge_vocabulary(*vocabularies.to_delayed()))
            vocabulary_for_transform = vocabulary_for_transform.persist()
            vocabulary_ = vocabulary.compute()
            n_features = len(vocabulary_)

        meta = scipy.sparse.csr_matrix((0, n_features), dtype=self.dtype)
        if isinstance(raw_documents, dd.Series):
            result = raw_documents.map_partitions(
                _count_vectorizer_transform, vocabulary_for_transform,
                params, meta=meta)
        else:
            result = raw_documents.map_partitions(
                _count_vectorizer_transform, vocabulary_for_transform, params)
            result = build_array(result, n_features, meta)

        self.vocabulary_ = vocabulary_
        self.fixed_vocabulary_ = fixed_vocabulary

        return result

    def transform(self, raw_documents):
        params = self.get_CountVectorizer_params()
        vocabulary = params.pop("vocabulary")

        if vocabulary is None:
            check_is_fitted(self, "vocabulary_")
            vocabulary = self.vocabulary_

        if isinstance(vocabulary, dict):
            # scatter for the user
            try:
                client = get_client()
            except ValueError:
                vocabulary_for_transform = dask.delayed(vocabulary)
            else:
                (vocabulary_for_transform,) = client.scatter((vocabulary,),
                                                             broadcast=True)
        else:
            vocabulary_for_transform = vocabulary

        n_features = vocabulary_length(vocabulary_for_transform)
        meta = scipy.sparse.csr_matrix((0, n_features), dtype=self.dtype)
        if isinstance(raw_documents, dd.Series):
            result = raw_documents.map_partitions(
                _count_vectorizer_transform, vocabulary_for_transform,
                params, meta=meta)
        else:
            transformed = raw_documents.map_partitions(
                _count_vectorizer_transform, vocabulary_for_transform, params)
            result = build_array(transformed, n_features, meta)
        return result

class TfidfTransformer(sklearn.feature_extraction.text.TfidfTransformer):
    """Transform a count matrix to a normalized tf or tf-idf representation

    See Also
    --------
    sklearn.feature_extraction.text.TfidfTransformer

    Examples
    --------
    >>> from dask_ml.feature_extraction.text import TfidfTransformer
    >>> from dask_ml.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> import numpy as np
    >>> corpus = ['this is the first document',
    ...           'this document is the second document',
    ...           'and this is the third one',
    ...           'is this the first document']
    >>> X = CountVectorizer().fit_transform(corpus)
    dask.array<concatenate, shape=(nan, 9), dtype=int64, chunksize=(nan, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
           [0, 2, 0, 1, 0, 1, 1, 0, 1],
           [1, 0, 0, 1, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 0, 0, 1, 0, 1]])
    >>> transformer = TfidfTransformer().fit(X)
    TfidfTransformer()
    >>> transformer.idf_
    array([1.91629073, 1.22314355, 1.51082562, 1.        , 1.91629073,
           1.91629073, 1.        , 1.91629073, 1.        ])
    >>> transformer.transform(X).compute().shape
    (4, 9)
    """
    def fit(self, X, y=None):
        """Learn the idf vector (global term weights).

        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        """
        def get_idf_diag(X, dtype):
            n_samples = _n_samples(X)   # X.shape[0] is not yet known
            n_features = X.shape[1]
            df = _document_frequency(X, dtype)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            return np.log(n_samples / df) + 1

        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            self._idf_diag = get_idf_diag(X, dtype)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts

        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
        """
        # X = self._validate_data(
        #     X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy, reset=False
        # )
        # if not sp.issparse(X):
        #     X = sp.csr_matrix(X, dtype=np.float64)

        def _astype(chunk, Xdtype=np.float64):
            return chunk.astype(Xdtype, copy=True)

        def _one_plus_log(chunk):
            # transforms nonzero elements x of csr_matrix: x -> 1 + log(x)
            c = chunk.copy()
            c.data = np.log(chunk.data, dtype=chunk.data.dtype)
            c.data += 1
            return c

        def _dot_idf_diag(chunk):
            return chunk * self._idf_diag

        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
        meta = scipy.sparse.eye(0, format="csr", dtype=dtype)
        if X.dtype != dtype:
            X = X.map_blocks(_astype, Xdtype=dtype, dtype=dtype, meta=meta)

        if self.sublinear_tf:
            X = X.map_blocks(_one_plus_log, dtype=dtype, meta=meta)

        if self.use_idf:
            # idf_ being a property, the automatic attributes detection
            # does not work as usual and we need to specify the attribute
            # name:
            check_is_fitted(self, attributes=["idf_"],
                            msg="idf vector is not fitted")
            self.__compute_idf()
            X = X.map_blocks(_dot_idf_diag, dtype=dtype, meta=meta)

        if self.norm:
            X = X.map_blocks(_normalize_transform,
                             dtype=dtype,
                             norm=self.norm,
                             meta=meta)

        return X

    def __compute_idf(self):
        # if _idf_diag is still lazy, then it is computed here
        if dask.is_dask_collection(self._idf_diag):
            _idf_diag = self._idf_diag.compute()
            n_features = len(_idf_diag)
            self._idf_diag = scipy.sparse.diags(
                _idf_diag,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=_idf_diag.dtype)
        
    @property
    def idf_(self):
        """Inverse document frequency vector, only defined if `use_idf=True`.

        Returns
        -------
        ndarray of shape (n_features,)
        """
        self.__compute_idf()
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))


class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer,
                      CountVectorizer):
    r"""Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TfidfTransformer`.

    See Also
    --------
    sklearn.feature_extraction.text.TfidfVectorizer

    Examples
    --------
    The Dask-ML implementation currently requires that ``raw_documents``
    is either a :class:`dask.bag.Bag` of documents (lists of strings) or
    a :class:`dask.dataframe.Series` of documents (Series of strings)
    with partitions of type :class:`pandas.Series`.

    >>> from dask_ml.feature_extraction.text import TfidfVectorizer
    >>> import dask.bag as db
    >>> from distributed import Client
    >>> client = Client()
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> corpus_bag = db.from_sequence(corpus, npartitions=2)
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus_bag)
    dask.array<concatenate, shape=(4, 9), dtype=float64, chunksize=(2, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524],
       [0.        , 0.6876236 , 0.        , 0.28108867, 0.        ,
        0.53864762, 0.28108867, 0.        , 0.28108867],
       [0.51184851, 0.        , 0.        , 0.26710379, 0.51184851,
        0.        , 0.26710379, 0.51184851, 0.26710379],
       [0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524]])
    >>> vectorizer.get_feature_names()
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> corpus_dds = dd.from_pandas(pd.Series(corpus), npartitions=2)
    >>> vectorizer = TfidfVectorizer()
    >>> X = vectorizer.fit_transform(corpus_dds)
    dask.array<concatenate, shape=(4, 9), dtype=float64, chunksize=(2, 9), ...
               chunktype=scipy.csr_matrix>
    >>> X.compute().toarray()
    array([[0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524],
       [0.        , 0.6876236 , 0.        , 0.28108867, 0.        ,
        0.53864762, 0.28108867, 0.        , 0.28108867],
       [0.51184851, 0.        , 0.        , 0.26710379, 0.51184851,
        0.        , 0.26710379, 0.51184851, 0.26710379],
       [0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524]])
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
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):

        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

        self._tfidf = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf
        )


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
    elif isinstance(vocabulary, Delayed):
        try:
            return len(vocabulary)
        except TypeError:
            return len(vocabulary.compute())
    elif isinstance(vocabulary, distributed.Future):
        client = get_client()
        future = client.submit(len, vocabulary)
        wait(future)
        result = future.result()
        return result
    else:
        raise ValueError(f"Unknown vocabulary type {type(vocabulary)}.")


def _normalize_transform(chunk, norm):
    return sklearn.preprocessing.normalize(chunk, norm=norm)


def _count_vectorizer_transform(partition, vocabulary, params):
    model = sklearn.feature_extraction.text.CountVectorizer(
        vocabulary=vocabulary, **params
    )
    return model.transform(partition)


def _build_vocabulary(partition, params):
    model = sklearn.feature_extraction.text.CountVectorizer(**params)
    model.fit(partition)
    return set(model.vocabulary_)


@dask.delayed
def _merge_vocabulary(*vocabularies):
    vocabulary = {
        key: i
        for i, key in enumerate(
            sorted(set(itertools.chain.from_iterable(vocabularies)))
        )
    }
    return vocabulary
