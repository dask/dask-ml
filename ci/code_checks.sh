#!/bin/bash

source activate dask-ml-test
MSG='Checking flake8... ' ; echo $MSG
flake8
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking black... ' ; echo $MSG
black --check .
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking isort... ' ; echo $MSG
isort --recursive --check-only .
RET=$(($RET + $?)) ; echo $MSG "DONE"

MSG='Checking coverage... ' ; echo $MSG
bash <(curl -s https://codecov.io/bash)
RET=$(($RET + $?)) ; echo $MSG "DONE"

exit $RET
