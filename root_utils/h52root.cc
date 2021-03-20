#include "H5Cpp.h"
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"

#include <string>
#include <iostream>
#include <stdexcept>
#include <cassert>

using namespace H5;
using namespace std;

const int maxentry = 10;

class Reader {
public:
    Reader(H5File &fin, string name);
    ~Reader();
    void GetEntry(hsize_t i);
    hsize_t GetEntries() { return dims[0]; }
    TBranch *Branch(TTree *tree);
protected:
    string name;
    H5T_class_t type_class;
    int ndims;
    hsize_t dims[2];

    int *intBuffer;
    float *floatBuffer;
    char **stringBuffer;

    int **intAccessor;
    float **floatAccessor;
    char ***stringAccessor;

    int intEntry[maxentry];
    float floatEntry[maxentry];
    vector<string> stringEntry;

    void ReadInt1D  (DataSet &dataset);
    void ReadFloat1D(DataSet &dataset);
    void ReadFloat2D(DataSet &dataset);
    void ReadString1D  (DataSet &dataset);
};

Reader::Reader(H5File &fin, string name) : name(name), intBuffer(NULL), floatBuffer(NULL), stringBuffer(NULL), intAccessor(NULL), floatAccessor(NULL), stringAccessor(NULL), stringEntry(maxentry) {
    DataSet dataset = fin.openDataSet(name);
    type_class = dataset.getTypeClass();

    DataSpace dataspace = dataset.getSpace();
    ndims = dataspace.getSimpleExtentNdims();

    if (type_class == H5T_FLOAT && ndims == 2) {
        ReadFloat2D(dataset);
    }
    else if (type_class == H5T_FLOAT && ndims == 1) {
        ReadFloat1D(dataset);
    }
    else if (type_class == H5T_INTEGER && ndims == 1) {
        ReadInt1D(dataset);
    }
    else if (type_class == H5T_ENUM) {
        ReadInt1D(dataset);
    }
    else if (type_class == H5T_STRING && ndims == 1) {
        ReadString1D(dataset);
    }
    else {
        throw invalid_argument(Form("Don't know how to handle %s with ndims = %d, type_class = %d", name.c_str(), ndims, int(type_class)));
    }
}

void Reader::ReadFloat2D(DataSet &dataset) {
    DataSpace dataspace = dataset.getSpace();
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    assert(ndims == 2);
    assert(dims[1] < maxentry);

    floatBuffer = new float[dims[0]*dims[1]];
    floatAccessor = new float*[dims[0]];
    for (hsize_t i = 0; i < dims[0]; i++) {
        floatAccessor[i] = floatBuffer + (i*dims[1]);
    }

    dataset.read(floatBuffer, PredType::NATIVE_FLOAT);
}

void Reader::ReadFloat1D(DataSet &dataset) {
    DataSpace dataspace = dataset.getSpace();
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    assert(ndims == 1);

    dims[1] = 1;

    floatBuffer = new float[dims[0]];
    floatAccessor = new float*[dims[0]];
    for (hsize_t i = 0; i < dims[0]; i++) {
        floatAccessor[i] = floatBuffer + (i*dims[1]);
    }

    dataset.read(floatBuffer, PredType::NATIVE_FLOAT);
}

void Reader::ReadInt1D(DataSet &dataset) {
    DataSpace dataspace = dataset.getSpace();
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    assert(ndims == 1);

    dims[1] = 1;

    intBuffer = new int[dims[0]];
    intAccessor = new int*[dims[0]];
    for (hsize_t i = 0; i < dims[0]; i++) {
        intAccessor[i] = intBuffer + (i*dims[1]);
    }

    dataset.read(intBuffer, PredType::NATIVE_INT);
}

void Reader::ReadString1D(DataSet &dataset) {
    DataSpace dataspace = dataset.getSpace();
    int ndims = dataspace.getSimpleExtentDims(dims, NULL);
    assert(ndims == 1);

    dims[1] = 1;

    stringBuffer = new char*[dims[0]];
    stringAccessor = new char**[dims[0]];
    for (hsize_t i = 0; i < dims[0]; i++) {
        stringAccessor[i] = stringBuffer + (i*dims[1]);
    }

    // see https://support.hdfgroup.org/ftp/HDF5/examples/misc-examples/varlen.cpp
    DataType dtype = dataset.getDataType();
    dataset.read(stringBuffer, dtype);
}

void Reader::GetEntry(hsize_t i) {
    assert(i < dims[0]);
    if (type_class == H5T_FLOAT) {
        for (hsize_t j = 0; j < dims[1]; j++) {
            floatEntry[j] = floatAccessor[i][j];
        }
    }
    else if (type_class == H5T_INTEGER || type_class == H5T_ENUM) {
        for (hsize_t j = 0; j < dims[1]; j++) {
            intEntry[j] = intAccessor[i][j];
        }
    }
    else if (type_class == H5T_STRING) {
        for (hsize_t j = 0; j < dims[1]; j++) {
            stringEntry[j] = stringAccessor[i][j];
        }
    }
}

TBranch *Reader::Branch(TTree *tree) {
    if (type_class == H5T_FLOAT) {
        if (ndims == 2) {
            return tree->Branch(name.c_str(), floatEntry, Form("%s[%llu]/F", name.c_str(), dims[1]));
        }
        else {
            return tree->Branch(name.c_str(), floatEntry, Form("%s/F", name.c_str()));
        }
    }
    else if (type_class == H5T_INTEGER || type_class == H5T_ENUM) {
        if (ndims == 2) {
            return tree->Branch(name.c_str(), intEntry, Form("%s[%llu]/I", name.c_str(), dims[1]));
        }
        else {
            return tree->Branch(name.c_str(), intEntry, Form("%s/I", name.c_str()));
        }
    }
    else if (type_class == H5T_STRING) {
        if (ndims == 2) {
            return tree->Branch(name.c_str(), &stringEntry);
        }
        else {
            return tree->Branch(name.c_str(), &stringEntry[0]);
        }
    }
    else {
        throw invalid_argument(Form("Don't know what branch to create for %s", name.c_str()));
    }
    
    return NULL;
}

Reader::~Reader() {
    delete [] intBuffer;
    delete [] intAccessor;
    delete [] floatBuffer;
    delete [] floatAccessor;
    if (stringBuffer != NULL) {
        for (hsize_t i = 0; i < dims[0]; i++) {
            for (hsize_t j = 0; j < dims[1]; j++) {
                free(stringAccessor[i][j]);
            }
        }
    }
    delete [] stringBuffer;
    delete [] stringAccessor;
}

int main(int argc, char *argv[]) {
    // if (argc < 4) {
    //     printf("Usage: %s infile.h5 outfile.root key1 key2 ...\n", argv[0]);
    if (argc < 3) {
        printf("Usage: %s infile.h5 outfile.root\n", argv[0]);
        exit(1);
    }

    const char *finname = argv[1];
    const char *foutname = argv[2];

    H5File fin( finname, H5F_ACC_RDONLY );
    vector<Reader*> readers;

    TFile *fout = new TFile(foutname, "RECREATE");
    if (fout->IsZombie()) {
        printf("Could not create output file: %s\n", foutname);
        exit(2);
    }

    TTree *tree = new TTree("h5", Form("Converted from %s", finname));
    long long Nentries = -1;
    // for (int i = 3; i < argc; i++) {
    //     Reader *r = new Reader(fin, argv[i]);
    for (hsize_t i = 0; i < fin.getNumObjs(); i++) {
        char dsname[1024];
        fin.getObjnameByIdx(i, dsname, 1024);
        Reader *r = new Reader(fin, dsname);
        r->Branch(tree);

        if (Nentries < 0) {
            Nentries = r->GetEntries();
        }
        else {
            assert(r->GetEntries() == hsize_t(Nentries));
        }

        readers.push_back(r);
    }

    for (int ev = 0; ev < Nentries; ev++) {
        for (unsigned i = 0; i < readers.size(); i++) {
            readers.at(i)->GetEntry(ev);
        }
        tree->Fill();
    }

    tree->Write();
    fout->Close();

    cout << "Wrote to " << fout->GetName() << endl;

    for (unsigned i = 0; i < readers.size(); i++) {
        delete readers.at(i);
    }

    return 0;
}
