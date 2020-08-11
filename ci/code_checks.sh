#!/bin/bash

source activate dask-ml-test
MSG='Checking flake8... ' ; echo $MSG
flake8
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking black... ' ; echo $MSG
black --version
black --check .
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking isort... ' ; echo $MSG
isort --version-number
isort --recursive --check-only .
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking mypy... ' ; echo $MSG
mypy dask_ml/metrics
mypy dask_ml/preprocessing
RET=$(($RET + $?)) ; echo $MSG "DONE"

exit $RET
