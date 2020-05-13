#include <TFile.h>
#include <TMatrixD.h>
#include <stdlib.h>

int main() {
	TFile *fin = new TFile("/home/lukasb/watchmal/20190731-topoCNN-WCsim/grid_3M_IWCD_0pad.root");
	TMatrixD mGridPmt = *(TMatrixD *)fin->Get("mGridPmt");
	printf("mGridPmt(0,0) = %g\n", mGridPmt(0,0));
	delete fin;
}

