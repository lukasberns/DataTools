#!/bin/bash
export HDF5ROOT="${PWD}/../../hdf5-1.10.5"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HDF5ROOT}/src/.libs:${HDF5ROOT}/c++/src/.libs"

export WCSIMLIB="$HOME/watchmal/WCSim_build"
export WCSIMDIR="$HOME/watchmal/WCSim"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WCSIMLIB}"
