#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

export HDF5_PLUGIN_PATH=hdf5_1.10/h5plugin

prefix/bin/frameProcessor --ctrl=tcp://0.0.0.0:10044 --config=$SCRIPT_DIR/fp5.json --log-config $SCRIPT_DIR/log4cxx.xml
