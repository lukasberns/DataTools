#!/bin/bash
HDF5ROOT="${PWD}/../../hdf5-1.10.5"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HDF5ROOT}/src/.libs:${HDF5ROOT}/c++/src/.libs"
