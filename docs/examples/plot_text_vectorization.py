"""
.. _plot_text_vectorization.py:

Text Vectorization Example
==========================

This example illustrates how dask ml can be used to
vectorize textual data.
"""
import os
import re
import tarfile
from glob import glob

import scipy.sparse
from sklearn.datasets import get_data_home
from sklearn.externals.six.moves.urllib.request import urlretrieve

import dask
import dask.bag as db
import dask.multiprocessing

from dask_ml.feature_extraction.text import HashingVectorizer

dask.set_options(get=dask.multiprocessing.get)


# adapted from
# https://github.com/scikit-learn/scikit-learn/tree/master/examples/applications/plot_out_of_core_classification.py#L143
# some other document collection may be used instead e.g. 20 newsgoups, RCV1..

def fetch_reuters(data_path=None):
    """Fetch documents of the Reuters dataset.
    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb))

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urlretrieve(DOWNLOAD_URL, filename=archive_path,
                    reporthook=progress)
        print('\r')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")
    return data_path


def load_from_filename(file_path):
    with open(file_path, 'rb') as fh:
        txt = fh.read().decode('latin-1')

    return re.findall('(?<=<BODY>)[^<]+(?=</BODY>)', txt)


data_path = fetch_reuters()
files = glob(os.path.join(data_path, 'reut2*'))

text = (db.from_sequence(files)
          .map(load_from_filename)
          .flatten())

vect = HashingVectorizer()

X = vect.fit_transform(text)

# there should be a a way to convert X to a sparse dask array
# without moving & serializing the data from the respective processes
X = scipy.sparse.vstack(X.compute())
print("Result: ", type(X), "shape=", X.shape, "nnz=", X.nnz)
