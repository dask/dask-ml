import dask.dataframe as dd
from dask.distributed import Client


def main():
    client = Client()  # noqa

    categories = ["category_%d" % i for i in range(26)]
    columns = ["click"] + ["numeric_%d" % i for i in range(13)] + categories

    df = dd.read_csv("day_1", sep="\t", names=columns, header=None)

    encoding = {c: "bytes" for c in categories}
    fixed = {c: 8 for c in categories}
    df.to_parquet(
        "day-1-bytes.parquet",
        object_encoding=encoding,
        fixed_text=fixed,
        compression="SNAPPY",
    )


if __name__ == "__main__":
    main()
