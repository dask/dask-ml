#!/bin/bash

echo "[install-travis]"

# install iniconda
MINICONDA_DIR="$HOME/miniconda3"
time wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh || exit 1
time bash miniconda.sh -b -p "$MINICONDA_DIR" || exit 1

echo
echo "[show conda]"
which conda

echo
echo "[update conda]"
conda config --set always_yes true --set changeps1 false || exit 1
conda update -q conda

echo
echo "[conda build]"
conda install conda-build anaconda-client --yes

echo
echo "[add channels]"
conda config --add channels conda-forge || exit 1

conda create -q -n test-environment python=${PYTHON}
source activate test-environment

conda install -q \
      coverage \
      flake8 \
      pytest \
      pytest-cov \
      pytest-mock \
      mock \
      numpy \
      pandas \
      scikit-learn \

# development dask, distributed
pip install git+https://github.com/dask/dask
pip install git+https://github.com/dask/distributed

pip install -e .
conda list test-environment
exit 0
