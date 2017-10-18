apt-get update; apt-get install -y gcc g++
conda config --set always_yes true --set changeps1 false --set quiet true
conda update -q conda
conda install conda-build anaconda-client --yes
conda config --add channels conda-forge
conda env create -f ci/environment-${PYTHON}.yml --name=${ENV_NAME}
source activate ${ENV_NAME}
pip install --no-deps -e .
conda list ${ENV_NAME}
