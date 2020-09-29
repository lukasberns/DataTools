#include <vector>
#include <fstream>

using namespace std;

// these methods are transcribed from pos_utils.py
int module_index(int pmt_index) {
    // """Returns the module number given the 0-indexed pmt number"""
    return pmt_index/19;
}

int pmt_in_module_id(int pmt_index) {
    // """Returns the pmt number within a 
    // module given the 0-indexed pmt number"""
    return pmt_index%19;
}

void create_grid_opencan(const char *wcsimfile, const char *posfile, const char *outfile) {
	// posfile is a csv file with the row/column index for each module

	bool mPMTsAsLayers = true;
	gSystem->Load("$WCSIMLIB/libWCSimRoot.so");

	TFile *fin = new TFile(wcsimfile);
	if (fin->IsZombie()) {
		cerr << "cannot open infile: " << infile << endl;
		exit(6);
	}

	ifstream fpos(posfile);
	if (!fpos.is_open()) {
		cerr << "Error opening posfile: " << posfile << endl;
		exit(34);
	}

	TFile *fout = new TFile(outfile, "RECREATE");
	if (fout->IsZombie()) {
		cerr << "Could not open outfile: " << outfile << endl;
		exit(40);
	}

	const int NpmtMax = 20000;
	// pmt position [cm]
	Double_t x[NpmtMax];
	Double_t y[NpmtMax];
	Double_t z[NpmtMax];
	// pmt orientation
	Double_t nx[NpmtMax];
	Double_t ny[NpmtMax];
	Double_t nz[NpmtMax];

	Int_t numPMT;
	Int_t numPMTsInmPMT = 1;

	TTree *Geometry = (TTree *)fin->Get("Geometry");
	if (Geometry == NULL) {
		// not flat file
		WCSimRootGeom *geom = NULL;
		TTree *wcsimGeoT = (TTree *)fin->Get("wcsimGeoT");
		if (wcsimGeoT == NULL) {
			cerr << "Could not find treee with geometry" << endl;
			return;
		}
		if (wcsimGeoT->GetEntries() < 1) {
			cerr << "Not enough entries in geom tree" << endl;
			return;
		}
		wcsimGeoT->SetBranchAddress("wcsimrootgeom", &geom);
		wcsimGeoT->GetEntry(0);

		numPMT = geom->GetWCNumPMT();
		int firstmPMTNo = -1;
		int nummPMT = 0;
		for (int i = 0; i < numPMT; i++) {
			const WCSimRootPMT &pmt = geom->GetPMT(i);
			int idx = i;
			if (mPMTsAsLayers) {
				int numInmPMT = pmt.GetmPMT_PMTNo()-1; // make it 0-indexed
				if (numInmPMT+1 > numPMTsInmPMT) {
					numPMTsInmPMT = numInmPMT+1;
				}

				if (i == 0) {
					firstmPMTNo = pmt.GetmPMTNo();
				}
				idx = pmt.GetmPMTNo()-firstmPMTNo;
				if (idx+1 > nummPMT) {
					nummPMT = idx+1;
				}

				if (numInmPMT > 0) {
					// we will only store the first
					// especially important for setting the tubeId
					continue;
				}
			}
			x[idx] = pmt.GetPosition(0);
			y[idx] = pmt.GetPosition(1);
			z[idx] = pmt.GetPosition(2);
			nx[idx] = pmt.GetOrientation(0);
			ny[idx] = pmt.GetOrientation(1);
			nz[idx] = pmt.GetOrientation(2);
		}

		if (mPMTsAsLayers) {
			numPMT = nummPMT;
		}
	}
	else {
		Geometry->SetBranchAddress("numPMT_ID", &numPMT);

		Geometry->SetBranchAddress("x", x);
		Geometry->SetBranchAddress("y", y);
		Geometry->SetBranchAddress("z", z);
		Geometry->SetBranchAddress("direction_x", nx);
		Geometry->SetBranchAddress("direction_y", ny);
		Geometry->SetBranchAddress("direction_z", nz);

		Geometry->GetEntry(0);
	}


	// now read the pos file
	// it's a csv file with i,j
	int maxi = 0;
	int maxj = 0;
	int ipmt = 0;
	int pmti[NpmtMax];
	int pmtj[NpmtMax];
	while (!fpos.eof()) {
		int i;
		int j;
		char comma;
		const int MAXW = 128;
		fpos >> i >> comma >> j;
		if (fpos.fail()) {
			cerr << "Skipping line " << ipmt << " of pos file (ok at end of file)" << endl;
			break;
		}
		fpos.ignore(MAXW, '\n');
		pmti[ipmt] = i;
		pmtj[ipmt] = j;
		if (i > maxi) { maxi = i; }
		if (j > maxj) { maxj = j; }
		ipmt++;
	}
	fpos.close();
	if (ipmt != numPMT) {
		cerr << "Inconsistent number of pmts in pos file and wcsim geom" << endl;
		cerr << "pos file:   " << ipmt << endl;
		cerr << "wcsim geom: " << numPMT << endl;
		exit(152);
	}

	Int_t N = numPMT;
	Int_t M = TMath::Max(maxi+1, maxj+1);

	int gridI[NpmtMax];
	int gridJ[NpmtMax];
	int gridPmt[NpmtMax];
	const int isEmpty = -1;

	for (int i = 0; i < M; i++) {
	  for (int j = 0; j < M; j++) {
	    int idx = i*M + j;
	    gridI[idx] = i;
	    gridJ[idx] = j;
	    gridPmt[idx] = isEmpty;
	  }
	}

	TMatrixD mGridPmt (M,M); // technicially we should define this as Int, but should be ok for the number of pmts considered
	TMatrixD mGridX   (M,M);
	TMatrixD mGridY   (M,M);
	TMatrixD mGridZ   (M,M);
	TMatrixD mGridDirX(M,M);
	TMatrixD mGridDirY(M,M);
	TMatrixD mGridDirZ(M,M);

	mGridPmt = isEmpty;

	for (int ipmt = 0; ipmt < numPMT; ipmt++) {
		int i = pmti[ipmt];
		int j = pmtj[ipmt];
		mGridPmt(i,j) = ipmt;
		mGridX   (i,j) =  x[ipmt];
		mGridY   (i,j) =  y[ipmt];
		mGridZ   (i,j) =  z[ipmt];
		mGridDirX(i,j) = nx[ipmt];
		mGridDirY(i,j) = ny[ipmt];
		mGridDirZ(i,j) = nz[ipmt];
	}

	TVectorD vNumPMTLayers(1);
	vNumPMTLayers(0) = numPMTsInmPMT;
	vNumPMTLayers.Write("vNumPMTLayers");

	mGridPmt .Write("mGridPmt");
	mGridX   .Write("mGridX");
	mGridY   .Write("mGridY");
	mGridZ   .Write("mGridZ");
	mGridDirX.Write("mGridDirX");
	mGridDirY.Write("mGridDirY");
	mGridDirZ.Write("mGridDirZ");
	fout->Close();
	printf("Wrote to %s\n", fout->GetName());
}
