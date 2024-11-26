uv pip install -e .[dev]

# Optionally, install development versions of dependenies
if [[ ${UPSTREAM_DEV} ]]; then
    uv pip install --no-deps --pre \
        -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
        scikit-learn

    uv pip install \
        --upgrade \
        git+https://github.com/dask/dask \
        git+https://github.com/dask/distributed
fi

# Install dask-ml
uv pip install --no-deps -e .
uv pip tree
