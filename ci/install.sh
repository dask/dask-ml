
# Optionally, install development versions of dependenies
if [[ ${UPSTREAM_DEV} ]]; then
    # FIXME https://github.com/mamba-org/mamba/issues/412
    # mamba uninstall --force numpy pandas scikit-learn
    conda uninstall --force numpy pandas scikit-learn

    python -m pip install --no-deps --pre \
        -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
        numpy \
        pandas \
        scikit-learn

    python -m pip install \
        --upgrade \
        locket \
        git+https://github.com/pydata/sparse \
        git+https://github.com/dask/dask \
        git+https://github.com/dask/distributed
fi

# Install dask-ml
python -m pip install --quiet --no-deps -e .

echo mamba list
mamba list