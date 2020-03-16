"""
K-Means clustering for the NYC-taxis data.set
"""
import argparse
import logging
import sys
from timeit import default_timer as tic

import dask.array as da
import dask.dataframe as dd
import pandas as pd
from dask import persist
from distributed import Client

from dask_ml.cluster import KMeans
from dask_ml.utils import _timer

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--scheduler-address",
        default=None,
        help="Address for the scheduler node. Runs locally by " "default.",
    )
    parser.add_argument("--start", default=5, type=int, help="Lower bound for clusters")
    parser.add_argument("--stop", default=26, type=int, help="Upper bound for clusters")
    parser.add_argument("--step", default=5, type=int, help="Cluster step size")
    parser.add_argument(
        "-f", "--factor", default=2, type=int, help="Oversampling factor."
    )
    return parser.parse_args(args)


def read():
    df = dd.read_parquet("s3://dask-data/nyc-taxi/nyc-2015.parquet", index=False)
    return df


def transform(df):
    df = (
        df.assign(
            **dict(
                duration=(
                    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
                ).dt.total_seconds(),
                hour=df.tpep_pickup_datetime.dt.hour,
                store_and_fwd_flag=df.store_and_fwd_flag != "N",
            )
        )
        .drop(["tpep_pickup_datetime", "tpep_dropoff_datetime"], axis=1)
        .astype(float)
    )
    return df


def as_array(df):
    lengths = df.map_partitions(len).compute()
    X = df.values
    blocks = X.to_delayed().flatten()
    P = X.shape[1]
    arrays = [
        da.from_delayed(x, dtype=X.dtype, shape=(length, P))
        for x, length in zip(blocks, lengths)
    ]
    X2 = da.concatenate(arrays, axis=0)
    return X2


def do(X, n_clusters, factor):
    km = KMeans(n_clusters=n_clusters, oversampling_factor=factor)
    km.fit(X)
    return km


def main(args=None):
    args = parse_args(args)
    steps = range(args.start, args.stop, args.step)
    if args.scheduler_address:
        client = Client(args.scheduler_address)
        info = client.scheduler_info()
        logger.info("Distributed mode: %s", client.scheduler)
        logger.info("Dashboard: %s:%s", info["address"], info["services"]["bokeh"])
    else:
        logger.warning("Local mode")

    logger.info("Fitting for %s", list(steps))

    logger.info("Reading data")
    X = read().pipe(transform).pipe(as_array)
    (X,) = persist(X)

    timings = []

    for n_clusters in range(args.start, args.stop, args.step):
        logger.info("Starting %02d", n_clusters)
        t0 = tic()
        with _timer(n_clusters, _logger=logger):
            km = do(X, n_clusters, factor=args.factor)
        t1 = tic()
        logger.info("Cluster Centers [%s]:\n%s", n_clusters, km.cluster_centers_)
        inertia = km.inertia_.compute()
        logger.info("Inertia [%s]: %s", km.cluster_centers_, inertia)
        timings.append((n_clusters, args.factor, t1 - t0, inertia))

    pd.DataFrame(timings, columns=["n_clusters", "factor", "time", "inertia"]).to_csv(
        "timings.csv"
    )


if __name__ == "__main__":
    sys.exit(main(None))
