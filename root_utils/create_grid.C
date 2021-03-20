#define WITH_MPMT 0

void create_grid(const char *infile, const char *outfile, bool yIsTankAxis = false, bool mPMTsAsLayers = false) {
	// set yIsTankAxis for nuPRISM tank
	// if false, z is the tank axis
	//
	// set mPMT to true to store each small PMT inside an mPMT as its own layer

	gSystem->Load("$WCSIMLIB/libWCSimRoot.so");

	TFile *fin = new TFile(infile);
	if (fin->IsZombie()) {
		cerr << "cannot open infile: " << infile << endl;
		exit(6);
	}

	TFile *fout = new TFile(outfile, "RECREATE");
	if (fout->IsZombie()) {
		cerr << "Could not open outfile: " << outfile << endl;
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
#if WITH_MPMT
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
#endif
			x[idx] = pmt.GetPosition(0);
			y[idx] = pmt.GetPosition(1);
			z[idx] = pmt.GetPosition(2);
			nx[idx] = pmt.GetOrientation(0);
			ny[idx] = pmt.GetOrientation(1);
			nz[idx] = pmt.GetOrientation(2);
		}

#if WITH_MPMT
		if (mPMTsAsLayers) {
			numPMT = nummPMT;
		}
#endif
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


	// additional values
	double rho[NpmtMax];
	double phi[NpmtMax];
	// 2d grid position
	Double_t X[NpmtMax];
	Double_t Y[NpmtMax];

	double *h = z;
	for (int ipmt = 0; ipmt < numPMT; ipmt++) {
	  if (yIsTankAxis) { // x -> z -> y -> x
	    rho[ipmt] = sqrt(z[ipmt]*z[ipmt] + x[ipmt]*x[ipmt]);
	    phi[ipmt] = atan2(x[ipmt],z[ipmt]);
	    h = y;
	  }
	  else {
	    rho[ipmt] = sqrt(x[ipmt]*x[ipmt] + y[ipmt]*y[ipmt]);
	    phi[ipmt] = atan2(y[ipmt],x[ipmt]);
	  }
	}

	// dimensions in [cm]
	Double_t tank_height = TMath::MaxElement(numPMT, h)-TMath::MinElement(numPMT, h);
	Double_t tank_radius = TMath::MaxElement(numPMT, rho);

	for (int ipmt = 0; ipmt < numPMT; ipmt++) {
	  double maxW = tank_radius*tank_radius + tank_radius*tank_height;
	  double W = sqrt((rho[ipmt]*rho[ipmt] - 2.*tank_radius*fabs(h[ipmt]) + tank_radius*tank_height)/(maxW));
	  
	  double X1 = W*(TMath::Pi() + phi[ipmt])/(2. * TMath::Pi());
	  double Y1 = W*(TMath::Pi() - phi[ipmt])/(2. * TMath::Pi());
	  
	  X[ipmt] = X1;
	  Y[ipmt] = Y1;
	  if (h[ipmt] > 0.) {
	    X[ipmt] = 1. - Y1;
	    Y[ipmt] = 1. - X1;
	  }
	}

	Int_t N = numPMT;
	Int_t gridPadding = 0; // extra space to help arranging
	Int_t M = TMath::CeilNint(sqrt(N)) + gridPadding;

	TRandom3 rndm(4357);

	double importance_zrho = 10.;
	double sortKey[NpmtMax];
	for (int ipmt = 0; ipmt < numPMT; ipmt++) {
	  sortKey[ipmt] = importance_zrho*(fabs(h[ipmt])/tank_height - rho[ipmt]) + rndm.Rndm();
	}
	Int_t sortIndex[NpmtMax];
	TMath::Sort(numPMT, sortKey, sortIndex, false);

	int gridI[NpmtMax];
	int gridJ[NpmtMax];
	double gridX[NpmtMax];
	double gridY[NpmtMax];

	int gridPmt[NpmtMax];
	const int isEmpty = -1;

	for (int i = 0; i < M; i++) {
	  for (int j = 0; j < M; j++) {
	    int idx = i*M + j;
	    gridI[idx] = i;
	    gridJ[idx] = j;
	    gridX[idx] = double(i)/double(M-1);
	    gridY[idx] = double(j)/double(M-1);
	    gridPmt[idx] = isEmpty;
	  }
	}

	double gridSortKey[NpmtMax];
	Int_t gridSortIndex[NpmtMax];

	for (int thispmt = 0; thispmt < numPMT; thispmt++) {
	  if (thispmt % 500 == 0) { printf("%d/%d\n", thispmt, numPMT); }
	  
	  for (int nextgrid = 0; nextgrid < M*M; nextgrid++) {
	    double dX = gridX[nextgrid] - X[sortIndex[thispmt]];
	    double dY = gridY[nextgrid] - Y[sortIndex[thispmt]];
	    double dW = dX+dY;
	    double dP = dX-dY;
	    double importance_dP = 3.; // we are more confident in the P~phi position than W~(h,rho)
	    double dist2 = dW*dW + importance_dP * dP*dP;
	    gridSortKey[nextgrid] = dist2;
	  }
	  TMath::Sort(M*M, gridSortKey, gridSortIndex, false);
	  for (int nextgrid = 0; nextgrid < M*M; nextgrid++) {
	    if (gridPmt[gridSortIndex[nextgrid]] != isEmpty) { continue; }
	    
	    gridPmt[gridSortIndex[nextgrid]] = sortIndex[thispmt];
	    break;
	  }
	}

	TMatrixD mGridPmt (M,M); // technicially we should define this as Int, but should be ok for the number of pmts considered
	TMatrixD mGridX   (M,M);
	TMatrixD mGridY   (M,M);
	TMatrixD mGridZ   (M,M);
	TMatrixD mGridDirX(M,M);
	TMatrixD mGridDirY(M,M);
	TMatrixD mGridDirZ(M,M);

	for (int i = 0; i < M; i++) {
	  for (int j = 0; j < M; j++) {
	    int idx = i*M + j;
	    mGridPmt (i,j) =    gridPmt[idx];
	    if (gridPmt[idx] == isEmpty) { continue; }
	    mGridX   (i,j) =  x[gridPmt[idx]];
	    mGridY   (i,j) =  y[gridPmt[idx]];
	    mGridZ   (i,j) =  z[gridPmt[idx]];
	    mGridDirX(i,j) = nx[gridPmt[idx]];
	    mGridDirY(i,j) = ny[gridPmt[idx]];
	    mGridDirZ(i,j) = nz[gridPmt[idx]];
	  }
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
