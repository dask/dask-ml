set -xe

apt-get update; apt-get install -y gcc g++
conda config --set always_yes true --set changeps1 false --set quiet true
conda update -q conda
conda install conda-build anaconda-client --yes --quiet
conda config --add channels conda-forge
conda env create -f ci/environment-${PYTHON}.yml --name=${ENV_NAME} --quiet
conda env list
source activate ${ENV_NAME}
pip install pip --upgrade
pip install --no-deps --quiet -e .
conda list -n ${ENV_NAME}
