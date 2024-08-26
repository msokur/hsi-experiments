#!/bin/bash

## set here the absolut root repository path
REPOSITORY_ROOT=$(dirname $(dirname "$(readlink -f "${BASH_SOURCE}")"))

## export the absolut root repository path to the python paths
export export PYTHONPATH=$REPOSITORY_ROOT