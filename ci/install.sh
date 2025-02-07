
# Optionally, install development versions of dependenies
if [[ ${UPSTREAM_DEV} ]]; then
    # FIXME https://github.com/mamba-org/mamba/issues/412
    # mamba uninstall --force dask distributed scikit-learn
    conda uninstall --force dask distributed scikit-learn

    python -m pip install --no-deps --pre \
        -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
        scikit-learn

    python -m pip install \
        --upgrade \
        git+https://github.com/dask/dask \
        git+https://github.com/dask/distributed
fi

# Install dask-ml
python -m pip install --quiet --no-deps -e .

echo mamba list
mamba list