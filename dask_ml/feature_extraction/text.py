import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import sklearn.feature_extraction.text


class HashingVectorizer(sklearn.feature_extraction.text.HashingVectorizer):
    def transform(self, X):
        """Transform a sequence of documents to a document-term matrix.

        Transformation is done in parallel, and correctly handles dask
        collections.

        Parameters
        ----------
        X : dask.Bag of raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
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
        msg = "'X' should be a 1-dimensional array with length 'num_samples'."

        if not dask.is_dask_collection(X):
            return super(HashingVectorizer, self).transform(X)

        if isinstance(X, db.Bag):
            bag2 = X.map_partitions(_transform, estimator=self)
            objs = bag2.to_delayed()
            arrs = [
                da.from_delayed(obj, (np.nan, self.n_features), self.dtype)
                for obj in objs
            ]
            result = da.concatenate(arrs, axis=0)
        elif isinstance(X, dd.Series):
            result = X.map_partitions(_transform, self)
        elif isinstance(X, da.Array):
            # dask.Array
            chunks = ((np.nan,) * X.numblocks[0], (self.n_features,))
            if X.ndim == 1:
                result = X.map_blocks(
                    _transform, estimator=self, dtype="f8", chunks=chunks, new_axis=1
                )
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

        return result


def _transform(part, estimator):
    return sklearn.feature_extraction.text.HashingVectorizer.transform(estimator, part)
