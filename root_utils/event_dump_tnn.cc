#include "H5Cpp.h"
#include "TFile.h"
#include "TTree.h"
#include "TMatrixD.h"

#include <string>
#include <cassert>

using namespace H5;
using namespace std;

void createDatasetInt(H5File &fout, const char *name, const TMatrixD &mat) {
	const int rank = 2;

	hsize_t dimsf[rank];
	dimsf[0] = mat.GetNrows();
	dimsf[1] = mat.GetNcols();
	DataSpace dataspace(rank, dimsf);

	int *data = new int[dimsf[0]*dimsf[1]]; // this ensures all values are consequtive, and should be passed to hdf5
	int **dataArr = new int*[dimsf[0]]; // this is for easy read/write in c++
	for (unsigned int i = 0; i < dimsf[0]; i++) {
		dataArr[i] = data + (i*dimsf[1]);
		for (unsigned int j = 0; j < dimsf[1]; j++) {
			dataArr[i][j] = mat(i,j);
		}
	}

	IntType datatype(PredType::NATIVE_INT);
	datatype.setOrder(H5T_ORDER_LE);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_INT);

	delete [] data;
	delete [] dataArr;;
}

void createDataset(H5File &fout, const char *name, const TMatrixD &mat) {
	const int rank = 2;

	hsize_t dimsf[rank];
	dimsf[0] = mat.GetNrows();
	dimsf[1] = mat.GetNcols();
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_DOUBLE);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(mat.GetMatrixArray(), PredType::NATIVE_DOUBLE);
}

void createDataset(H5File &fout, const char *name, const float *data, int n1) {
	const int rank = 1;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_FLOAT);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_FLOAT);
}

void createDataset(H5File &fout, const char *name, const float *data, int n1, int n2) {
	const int rank = 2;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	dimsf[1] = n2;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_FLOAT);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_FLOAT);
}

void createDataset(H5File &fout, const char *name, const float *data, int n1, int n2, int n3) {
	const int rank = 3;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	dimsf[1] = n2;
	dimsf[2] = n3;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_FLOAT);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_FLOAT);
}

void createDataset(H5File &fout, const char *name, const float *data, int n1, int n2, int n3, int n4) {
	const int rank = 4;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	dimsf[1] = n2;
	dimsf[2] = n3;
	dimsf[3] = n4;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_FLOAT);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_FLOAT);
}

void createDataset(H5File &fout, const char *name, const int *data, int n1) {
	const int rank = 1;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_INT);
	datatype.setOrder(H5T_ORDER_LE);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_INT);
}

void createDataset(H5File &fout, const char *name, const int *data, int n1, int n2) {
	const int rank = 2;

	hsize_t dimsf[rank];
	dimsf[0] = n1;
	dimsf[1] = n2;
	DataSpace dataspace(rank, dimsf);

	IntType datatype(PredType::NATIVE_INT);
	datatype.setOrder(H5T_ORDER_LE);

	DataSet dataset = fout.createDataSet(name, datatype, dataspace);
	dataset.write(data, PredType::NATIVE_INT);
}

int getLabel(string infile) {
	if (infile.find("_gamma_") != string::npos) {
		return 0;
	}
	if (infile.find("_e-_") != string::npos || infile.find("_e+_") != string::npos) {
		return 1;
	}
	if (infile.find("_mu-_") != string::npos || infile.find("_mu+_") != string::npos) {
		return 2;
	}
	if (infile.find("_pi0_") != string::npos) {
		return 3;
	}
	printf("Error: Unknown input file particle type.\n");
	exit(150);
}

int main(int argc, char *argv[]) {
	if (argc < 5) {
		printf("Usage: %s gridfile.root wcsimfile.root outfile.h5 label\n", argv[0]);
		printf("\n");
		printf("[label] 0:gamma, 1:electron, 2:muon, 3:pi0, 4:pi+/-\n");
		exit(1);
	}

	const char *fgridname = argv[1];
	const char *finname = argv[2];
	const char *foutname = argv[3];
	int label = atoi(argv[4]);

	TFile *fgrid = new TFile(fgridname);
	if (fgrid->IsZombie()) {
		printf("Could not open gridfile: %s\n", fgridname);
		return 12;
	}

	TFile *fin = new TFile(finname);
	if (fin->IsZombie()) {
		printf("Could not open wcsimfile (input): %s\n", finname);
		return 64;
	}

	TMatrixD *mGridPmtPtr = (TMatrixD *)fgrid->Get("mGridPmt");
	if (!mGridPmtPtr) {
		printf("Error: Could not find mGridPmt matrix in grid file.\n");
		exit(184);
	}
	TMatrixD mGridPmt  = *mGridPmtPtr;
	TMatrixD mGridX    = *(TMatrixD *)fgrid->Get("mGridX");
	TMatrixD mGridY    = *(TMatrixD *)fgrid->Get("mGridY");
	TMatrixD mGridZ    = *(TMatrixD *)fgrid->Get("mGridZ");
	TMatrixD mGridDirX = *(TMatrixD *)fgrid->Get("mGridDirX");
	TMatrixD mGridDirY = *(TMatrixD *)fgrid->Get("mGridDirY");
	TMatrixD mGridDirZ = *(TMatrixD *)fgrid->Get("mGridDirZ");

	H5File fout(foutname, H5F_ACC_TRUNC);
	createDatasetInt(fout, "mGridPmt",  mGridPmt);
  	createDataset   (fout, "mGridX",    mGridX);
  	createDataset   (fout, "mGridY",    mGridY);
  	createDataset   (fout, "mGridZ",    mGridZ);
  	createDataset   (fout, "mGridDirX", mGridDirX);
  	createDataset   (fout, "mGridDirY", mGridDirY);
  	createDataset   (fout, "mGridDirZ", mGridDirZ);
  
	TTree *Trigger = (TTree *)fin->Get("Trigger");
	if (!Trigger) {
		printf("Could not find Trigger tree in wcsim file. Make sure you pass the _flat.root file.\n");
		exit(201);
	}
	Int_t Event;
	Double_t StartTime;
	Trigger->SetBranchAddress("Event", &Event);
	Trigger->SetBranchAddress("StartTime", &StartTime);
	Long_t Nentries = Trigger->GetEntries();
	Trigger->GetEntry(Nentries-1);
	int Nevents = Event; // because Event starts at 1, this is number of events

	int *subEventEntry = new int[Nevents];
	double *subEventStartTime = new double[Nevents];
	for (Long_t ev = 0; ev < Nevents; ev++) {
		subEventEntry[ev] = -1;
		subEventStartTime[ev] = 0.;
	}

	for (Long_t ie = 0; ie < Nentries; ie++) {
		Trigger->GetEntry(ie);
		assert(Event > 0);
		assert(Event <= Nevents);
		if (subEventEntry[Event-1] == -1 || StartTime < subEventStartTime[Event-1]) {
			// new event, or earlier subevent
			subEventEntry[Event-1] = ie;
			subEventStartTime[Event-1] = StartTime;
		}
	}

	int Nrows = mGridPmt.GetNrows();
	int Ncols = mGridPmt.GetNcols();
	
	int *pmtGridRow = new int[Nrows*Ncols];
	int *pmtGridCol = new int[Nrows*Ncols];
	const int gridIsEmpty = -1;
	for (int i = 0; i < Nrows; i++) {
		for (int j = 0; j < Ncols; j++) {
			int itube = mGridPmt(i,j);
			if (itube == gridIsEmpty) { continue; }
			assert(itube >= 0);
			assert(itube < Nrows*Ncols);
			pmtGridRow[itube] = i;
			pmtGridCol[itube] = j;
		}
	}

	int Nchs = 2; // Q,T
	int *event_ids = new int[Nevents];
	int *labels       = new int[Nevents];
	float *event_data = new float[Nevents*Nrows*Ncols*Nchs];

	int NmaxOutTracks = 2;
	int *pids         = new int[Nevents*NmaxOutTracks];
	float *positions  = new float[Nevents*NmaxOutTracks*3];
	float *directions = new float[Nevents*NmaxOutTracks*3];
	float *energies   = new float[Nevents*NmaxOutTracks];

	TTree *CherenkovDigiHits = (TTree *)fin->Get("CherenkovDigiHits");
	Int_t NDigiHits;
	Int_t *Tube = new Int_t[Nrows*Ncols];
	Float_t *Q = new Float_t[Nrows*Ncols];
	Float_t *T = new Float_t[Nrows*Ncols];
	CherenkovDigiHits->SetBranchAddress("NDigiHits", &NDigiHits);
	CherenkovDigiHits->SetBranchAddress("Tube", Tube);
	CherenkovDigiHits->SetBranchAddress("Q", Q);
	CherenkovDigiHits->SetBranchAddress("T", T);

	TTree *Tracks = (TTree *)fin->Get("Tracks");
	Int_t Ntracks;
	const int NmaxTracks = 1000;
	Int_t ParentID[NmaxTracks];
	Int_t Pid[NmaxTracks];
	Int_t Flag[NmaxTracks];
	Float_t Energy[NmaxTracks];
	Float_t Dirx[NmaxTracks];
	Float_t Diry[NmaxTracks];
	Float_t Dirz[NmaxTracks];
	Float_t Start_x[NmaxTracks];
	Float_t Start_y[NmaxTracks];
	Float_t Start_z[NmaxTracks];
	Tracks->SetBranchAddress("Ntracks", &Ntracks);
	Tracks->SetBranchAddress("ParentID", ParentID);
	Tracks->SetBranchAddress("Pid", Pid);
	Tracks->SetBranchAddress("Flag", Flag);
	Tracks->SetBranchAddress("Energy", Energy);
	Tracks->SetBranchAddress("Dirx", Dirx);
	Tracks->SetBranchAddress("Diry", Diry);
	Tracks->SetBranchAddress("Dirz", Dirz);
	Tracks->SetBranchAddress("Start_x", Start_x);
	Tracks->SetBranchAddress("Start_y", Start_y);
	Tracks->SetBranchAddress("Start_z", Start_z);

	for (int ev = 0; ev < Nevents; ev++) {
		CherenkovDigiHits->GetEntry(subEventEntry[ev]);
		Tracks           ->GetEntry(subEventEntry[ev]);

		// initialize
		for (int i = 0; i < Nrows; i++) {
			for (int j = 0; j < Nrows; j++) {
				for (int c = 0; c < Nchs; c++) {
					int idx = ev*Nrows*Ncols*Nchs + i*Ncols*Nchs + j*Nchs + c;
					event_data[idx] = 0.;
				}
			}
		}
		for (int it = 0; it < NmaxOutTracks; it++) {
			int i = ev*NmaxOutTracks + it;
			pids    [i] = 0;
			energies[i] = 0.;
			positions[i*3+0] = 0.;
			positions[i*3+1] = 0.;
			positions[i*3+2] = 0.;
			directions[i*3+0] = 0.;
			directions[i*3+1] = 0.;
			directions[i*3+2] = 0.;
		}
		
		// get truth info
		int savedTracks = 0;
		for (int it = 0; it < Ntracks; it++) {
			if (ParentID[it] == 0 && Flag[it] == 0) {
				int i = ev*NmaxOutTracks + savedTracks;
				pids    [i] = Pid[it];
				energies[i] = Energy[it];
				positions[i*3+0] = Start_x[it];
				positions[i*3+1] = Start_y[it];
				positions[i*3+2] = Start_z[it];
				directions[i*3+0] = Dirx[it];
				directions[i*3+1] = Diry[it];
				directions[i*3+2] = Dirz[it];
				savedTracks++;
				assert(savedTracks < NmaxOutTracks);
			}
		}
		
		// get Q and T
		for (int ihit = 0; ihit < NDigiHits; ihit++) {
			// Tube starts at 1
			assert(Tube[ihit] >= 1);
			assert(Tube[ihit] <= Nrows*Ncols);
			int gridRow = pmtGridRow[Tube[ihit]-1];
			int gridCol = pmtGridCol[Tube[ihit]-1];
			int idx = ev*Nrows*Ncols*Nchs + gridRow*Ncols*Nchs + gridCol*Nchs;
			event_data[idx+0] = Q[ihit];
			event_data[idx+1] = T[ihit];
		}

		event_ids[ev] = ev;
		labels[ev] = label;
	}

	createDataset(fout, "event_data", event_data, Nevents, Nrows, Ncols, Nchs);
	createDataset(fout, "event_ids",  event_ids,  Nevents);
	createDataset(fout, "labels",     labels,     Nevents);
	createDataset(fout, "pids",       pids,       Nevents, NmaxOutTracks);
	createDataset(fout, "energies",   energies,   Nevents, NmaxOutTracks);
	createDataset(fout, "positions",  positions,  Nevents, NmaxOutTracks, 3);
	createDataset(fout, "directions", directions, Nevents, NmaxOutTracks, 3);

	printf("Wrote to %s\n", foutname);
	delete fgrid;
	delete [] subEventEntry;
	delete [] subEventStartTime;
	delete [] pmtGridRow;
	delete [] pmtGridCol;
	delete [] event_data;
	delete [] Tube;
	delete [] Q;
	delete [] T;
}
