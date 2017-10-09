"""
Script for running K-Means clustering on the KDD-Cup dataset.
"""
import argparse
import glob
import logging
import os
import string
import sys
from timeit import default_timer as tic

import dask.array as da
import dask.dataframe as dd
import pandas as pd
import requests
import s3fs
import sklearn.cluster as sk
from daskml.cluster import KMeans
from distributed import Client
from sklearn.datasets import get_data_home

logger = logging.getLogger()

try:
    import coloredlogs
except ImportError:
    pass
else:
    coloredlogs.install()

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz"  # noqa


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--scheduler-address', default=None,
                        help="Address for the scheduler node. Runs locally by "
                             "default.")
    parser.add_argument("-l", "--local", action='store_true',
                        help="Use local filesystem")
    parser.add_argument('--scikit-learn', action='store_true',
                        help="Just use scikit-learn")

    return parser.parse_args(args)


def download():
    p = os.path.join(get_data_home(), "kddcpu.data.gz")
    if os.path.exists(p):
        return p
    r = requests.get(URL, stream=True)
    with open(p, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return p


def split(p):
    output = os.path.join(get_data_home(), "kddcup.parq")
    if not os.path.exists(output):

        dtype = {
            1: 'category',
            2: 'category',
            3: 'category',
            41: 'category',
        }

        df = pd.read_csv(p, header=None, dtype=dtype)
        cat_cols = df.select_dtypes(include=['category']).columns
        df[cat_cols] = df[cat_cols].apply(lambda col: col.cat.codes)
        df.columns = list(string.ascii_letters[:len(df.columns)])

        ddf = dd.from_pandas(df, npartitions=16)
        ddf.to_parquet(output)

    return output


def upload(p, fs):
    fs.mkdir('dask-data/kddcup/kdd.parq')
    for file in glob(os.path.join(p, '*')):
        fs.put(file, 'dask-data/kddcup/kdd.parq/' + file.split('/')[-1])
        print(file)


def as_known(X, lengths):
    blocks = X.to_delayed().flatten()
    P = X.shape[1]
    arrays = [
        da.from_delayed(x, dtype=X.dtype, shape=(length, P))
        for x, length in zip(blocks, lengths)
    ]
    return da.concatenate(arrays, axis=0)


def load(path):
    logger.info("Reading data")
    df = dd.read_parquet(path)
    df = df.persist()
    logger.info("Data in memory")

    # Get known chunks
    lengths = df.map_partitions(len).compute()
    X = as_known(df.values, lengths)
    X = X.persist()
    return X


def fit(data, use_scikit_learn=False):
    logger.info("Starting to cluster")
    # Cluster
    n_clusters = 8
    oversampling_factor = 2
    if use_scikit_learn:
        km = sk.KMeans(n_clusters=n_clusters, random_state=0)
    else:
        km = KMeans(n_clusters=n_clusters,
                    oversampling_factor=oversampling_factor,
                    random_state=0)
    t0 = tic()
    logger.info("Starting n_clusters=%2d, oversampling_factor=%2d",
                n_clusters, oversampling_factor)
    km.fit(data)
    t1 = tic()
    logger.info("Finished in %.2f", t1 - t0)


def main(args=None):
    args = parse_args(args)
    logger.info("Checking local data")
    local = split(download())

    if args.scheduler_address:
        logger.info("Connecting to %s", args.scheduler_address)
        client = Client(args.scheduler_address)
        logger.info(client.scheduler_info())

    if not args.local:
        logger.info("Using distributed mode")
        fs = s3fs.S3FileSystem()
        if fs.exists("dask-data"):
            logger.info("Using cached dataset")
        else:
            logger.info("Uploading to cloud storage")
            upload(local, fs)
        path = "s3://dask-data/kddcup/kdd.parq/"
    else:
        logger.info("Using local mode")
        path = local

    data = load(path)
    fit(data, args.scikit_learn)


if __name__ == '__main__':
    sys.exit(main(None))
