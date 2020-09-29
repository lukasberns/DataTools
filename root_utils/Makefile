# HDF5ROOT = "../../hdf5-1.10.5"

CXXFLAGS = -I$(HDF5ROOT)/src -I$(HDF5ROOT)/c++/src -I$(WCSIMDIR)/include $(shell root-config --cflags)
LDFLAGS = -L$(HDF5ROOT)/src/.libs -L$(HDF5ROOT)/c++/src/.libs -L$(WCSIMLIB) -Wall
LDLIBS = -lhdf5 -lhdf5_cpp -lWCSimRoot $(shell root-config --libs)

all: h5test roottest event_dump_tnn h52root

h5test: h5test.cc
h52root: h52root.cc

roottest: roottest.cc

event_dump_tnn: event_dump_tnn.cc
