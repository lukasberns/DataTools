#include "TChain.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TVector3.h"

#include <cmath>
#include <cassert>
using namespace std;

const int nc = 2;
const int ic_fQ = 0;
const int ic_h5 = 1;

TH1D *hrelE[nc];
TH1D *hpos[nc];
TH1D *hposL[nc];
TH1D *hposT[nc];
TH1D *hdir[nc];

TH2D *hrelEtowall[nc];
TH2D *hpostowall[nc];
TH2D *hdirtowall[nc];

TH1D *hdwall;
TH1D *htowall;

const char *confst[nc];

void plotfQresnet(const char *recoversion, const char *recovershort) {

const char *pname = "e-";
// const char *pname = "mu-";

const char *datadir = "/home/lukasb/watchmal/data/IWCDmPMT_4pi_full_tank";
//const char *recoversion = "20200827-04-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir-n3d-364504";
//const char *recovershort = "ResNet 20200827-04-01";
//const char *recoversion = "20200827-04-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir-n3d-20200921-040028";
//const char *recovershort = "ResNet 20200827-04-01 0921-040";
//const char *recoversion = "20200827-04-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir-n3d-electron-20200921-043141";
//const char *recovershort = "ResNet 20200827-04 e- only";
//const char *recoversion = "20200827-05-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir-n3d-longtrans-20200922-041914";
//const char *recovershort = "ResNet 20200827-05 LT";
// const char *recoversion = "20200827-01-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir";
// const char *recovershort = "ResNet 20200827-01-01";

TChain *fQ = new TChain("fiTQun");
TChain *h5 = new TChain("h5");
const int bchOffset = 2;
const int NentriesPerFile = 3000;
for (int bch = bchOffset; bch < 100; bch++) {
// for (int bch = bchOffset; bch < 61; bch++) {
  fQ->Add(Form("%s/fiTQun/%s/E0to1000MeV/unif-pos-R371-y521cm/4pi-dir/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d_fiTQun.root", datadir, pname, pname, bch));
  h5->Add(Form("%s/reco_%s/%s/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d.root", datadir, recoversion, pname, pname, bch));
}

TChain *todveto = new TChain("h5");
todveto->Add(Form("%s/od_veto.root", datadir));

TChain *tmichels = new TChain("h5");
tmichels->Add(Form("%s/michels.root", datadir));


int fQ_nevt;
const int fiTQun_maxsubevt = 2;
const int fiTQun_npid = 7;
const int fiTQun_electron = 1;
const int fiTQun_muon     = 2;

float fq1rmom[fiTQun_maxsubevt][fiTQun_npid];
float fq1rpos[fiTQun_maxsubevt][fiTQun_npid][3];
float fq1rdir[fiTQun_maxsubevt][fiTQun_npid][3];
int fq1rpcflg[fiTQun_maxsubevt][fiTQun_npid];

fQ->SetBranchAddress("nevt", &fQ_nevt);
fQ->SetBranchAddress("fq1rmom", fq1rmom);
fQ->SetBranchAddress("fq1rpos", fq1rpos);
fQ->SetBranchAddress("fq1rdir", fq1rdir);
fQ->SetBranchAddress("fq1rpcflg", fq1rpcflg);

int true_label;
const int label_electron = 0;
const int label_muon = 1;
const int label_gamma = 2;
const int nlabel = 3;

const double mass[nlabel] = { 0.511, 105.7, 0.511*2. };
const double pthres[nlabel] = { 0.57, 118., 0.57 *2 };
double Ethres[nlabel];
for (int il = 0; il < nlabel; il++) {
  Ethres[il] = sqrt(pow(mass[il],2) + pow(pthres[il],2));
}

float true_Eabovethres;
float pred_Eabovethres;

float true_position[3];
float pred_position[3];

float true_direction[3];
float pred_direction[3];

h5->SetBranchAddress("true_label", &true_label);
h5->SetBranchAddress("true_Eabovethres", &true_Eabovethres);
h5->SetBranchAddress("pred_Eabovethres", &pred_Eabovethres);
h5->SetBranchAddress("true_positions", true_position);
h5->SetBranchAddress("pred_position" , pred_position);
//h5->SetBranchAddress("pred_simple_position" , pred_position);
h5->SetBranchAddress("true_directions", true_direction);
h5->SetBranchAddress("pred_direction" , pred_direction);

int od_veto;
int is_michel;
todveto->SetBranchAddress("veto", &od_veto);
tmichels->SetBranchAddress("is_michel", &is_michel);


int colors[nc] = { kAzure-2, kRed-4 };

confst[0] = "fiTQun";
confst[1] = recovershort;


hdwall = new TH1D("hdwall", ";Distance to wall [cm];Events", 200, 0., 520.);
htowall = new TH1D("htowall", ";Distance to wall along track [cm];Events", 200, 0., 520.);

for (int ic = 0; ic < nc; ic++) {
  hrelE[ic] = new TH1D(Form("hrelE_%d", ic), ";(pred - true)/true energy;Events", 200, -0.5, 0.5);
  hrelE[ic]->SetLineColor(colors[ic]);
  
  hpos[ic] = new TH1D(Form("hpos_%d", ic), ";Distance to true vertex [cm];Events", 200, 0., 200.);
  hpos[ic]->SetLineColor(colors[ic]);

  hposL[ic] = new TH1D(Form("hposL_%d", ic), ";Longitudinal distance to true vertex [cm];Events", 200, -200., 200.);
  hposL[ic]->SetLineColor(colors[ic]);

  hposT[ic] = new TH1D(Form("hposT_%d", ic), ";Transverse distance to true vertex [cm];Events", 200, 0., 200.);
  hposT[ic]->SetLineColor(colors[ic]);
  
  hdir[ic] = new TH1D(Form("hdir_%d", ic), ";Angle to true direction [deg];Events", 200, 0., 45.);
  hdir[ic]->SetLineColor(colors[ic]);


  hrelEtowall[ic] = new TH2D(Form("hrelEtowall_%d", ic), ";Distance to wall along track [cm];|pred - true|/true energy", 50, 0., 520., 200, 0., 0.5);
  hpostowall[ic] = new TH2D(Form("hpostowall_%d", ic), ";Distance to wall along track [cm];Distance to true vertex [cm]", 50, 0., 520., 200, 0., 200.);
  hdirtowall[ic] = new TH2D(Form("hdirtowall_%d", ic), ";Distance to wall along track [cm];Angle to true direction [deg]", 50, 0., 520., 200, 0., 45.);
}



Long_t fQ_Nentries = fQ->GetEntries();
Long64_t *h5TreeOffset = h5->GetTreeOffset();

h5->GetEntry(0);
assert(h5->GetTree()->GetEntries() == NentriesPerFile);

for (Long_t fQev = 0; fQev < fQ_Nentries; fQev++) {
  fQ->GetEntry(fQev);
  h5->GetEntry(h5TreeOffset[fQ->GetTreeNumber()] + fQ_nevt-1);
  
  int globalId = (bchOffset + fQ->GetTreeNumber())*NentriesPerFile + fQ_nevt-1;
  todveto->GetEntry(globalId);
  tmichels->GetEntry(globalId);
  
  double true_energy = true_Eabovethres + Ethres[true_label];
  double true_momentum = sqrt(pow(true_energy,2) - pow(mass[true_label],2) + 1e-16);
  
  double tankZ = 520.;
  double tankR = 370.;
  double dwall_caps = tankZ - fabs(true_position[1]);
  double dwall_barrel = tankR - sqrt(pow(true_position[0],2) + pow(true_position[2],2));
  double dwall = TMath::Min(dwall_caps,dwall_barrel);
  
  tankZ += 1.;
  tankR += 1.;
  // [x,y,z] = pos + t*dir
  // x^2 + y^2 = R^2
  // = (x0 + t*dx)^2 + (y0 + t*dy)^2
  // = x0^2 + 2t * x0*dx + t^2 * dx^2
  // 0 = t^2 + 2(x0.dx) t + (x0^2 - R^2)
  // t = - (x0.dx) + sqrt[(x0.dx)^2 - (x0^2 - R^2)]
  double exitpoint_barrel_x0dx = true_position[0]*true_direction[0] + true_position[2]*true_direction[2];
  double exitpoint_barrel_x0x0 = pow(true_position[0],2) + pow(true_position[2],2);
  double exitpoint_barrel_insqrt = pow(exitpoint_barrel_x0dx,2) + pow(tankR,2) - exitpoint_barrel_x0x0;
  double exitpoint_barrel_t = (exitpoint_barrel_insqrt>0. ? -exitpoint_barrel_x0dx + sqrt((exitpoint_barrel_insqrt>0. ? exitpoint_barrel_insqrt : 0.)) : NAN);
  double exitpoint_barrel[3];
  for (int i = 0; i < 3; i++) {
    exitpoint_barrel[i] = true_position[i] + exitpoint_barrel_t * true_direction[i];
  }
  
  // tankZ = z = pos + t*dir
  // t = (tankZ - pos_z)/dir_z
  double exitpoint_cap_t = ((true_direction[1]>0. ? tankZ : -tankZ) - true_position[1]) / true_direction[1];
  double exitpoint_cap[3];
  for (int i = 0; i < 3; i++) {
    exitpoint_cap[i] = true_position[i] + exitpoint_cap_t * true_direction[i];
  }
  
  double towall = ((TMath::IsNaN(exitpoint_barrel_t) || fabs((TMath::IsNaN(exitpoint_barrel_t) ? 0. : exitpoint_barrel[1])) > tankZ) ? exitpoint_cap_t : exitpoint_barrel_t);
  
  int fiTQun_pid;
  bool passCut = true;
  passCut &= dwall > 50.;
  if (true_label == label_muon) {
    passCut &= !od_veto;
    fiTQun_pid = fiTQun_muon;
    // fiTQun_pid = fiTQun_electron; /////////////////////// ############################### REMOVE
  }
  else {
    passCut &= (towall > 63.*log(true_momentum) - 200.);
    fiTQun_pid = fiTQun_electron;
  }
  passCut &= (fq1rpcflg[0][fiTQun_electron] == 0);
  passCut &= (fq1rpcflg[0][fiTQun_muon] == 0);

  if (!passCut) { continue; }
  //if (passCut) { continue; }
  
  double fq1rEabovethres = sqrt(pow(fq1rmom[0][fiTQun_pid],2) + pow(mass[true_label],2)) - Ethres[true_label];
  hrelE[ic_fQ]->Fill((fq1rEabovethres - true_Eabovethres)/true_Eabovethres);
  hrelE[ic_h5]->Fill((pred_Eabovethres - true_Eabovethres)/true_Eabovethres);
  
  
  for (int i = 0; i < 3; i++) {
    pred_position[i] *= 1000.;
  }
  
  TVector3 trpos(true_position);
  TVector3 fqpos(fq1rpos[0][fiTQun_pid]);
  TVector3 h5pos(pred_position);
  
  hpos[ic_fQ]->Fill((fqpos - trpos).Mag());
  hpos[ic_h5]->Fill((h5pos - trpos).Mag());

  TVector3 trdir(true_direction);

  hposL[ic_fQ]->Fill((fqpos - trpos) * trdir);
  hposL[ic_h5]->Fill((h5pos - trpos) * trdir);

  hposT[ic_fQ]->Fill((fqpos - trpos - ((fqpos - trpos) * trdir)*trdir).Mag());
  hposT[ic_h5]->Fill((h5pos - trpos - ((h5pos - trpos) * trdir)*trdir).Mag());

  
  TVector3 fqdir(fq1rdir[0][fiTQun_pid]);
  TVector3 h5dir(pred_direction);
  
  hdir[ic_fQ]->Fill(trdir.Angle(fqdir)/TMath::Pi()*180.);
  hdir[ic_h5]->Fill(trdir.Angle(h5dir)/TMath::Pi()*180.);
  
  htowall->Fill(towall);
  hdwall->Fill(dwall);

  hrelEtowall[ic_fQ]->Fill(towall, fabs(fq1rEabovethres - true_Eabovethres)/true_Eabovethres);
  hrelEtowall[ic_h5]->Fill(towall, fabs(pred_Eabovethres - true_Eabovethres)/true_Eabovethres);

  hpostowall[ic_fQ]->Fill(towall, (fqpos - trpos).Mag());
  hpostowall[ic_h5]->Fill(towall, (h5pos - trpos).Mag());

  hdirtowall[ic_fQ]->Fill(towall, trdir.Angle(fqdir)/TMath::Pi()*180.);
  hdirtowall[ic_h5]->Fill(towall, trdir.Angle(h5dir)/TMath::Pi()*180.);
}

TLegend *l1 = new TLegend(0.6, 0.65, 0.97, 0.85);
l1->SetBorderSize(0);

double ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hrelE[ic], confst[ic], "L");
  if (hrelE[ic]->GetMaximum() > ymax) {
    ymax = hrelE[ic]->GetMaximum();
  }
}
hrelE[ic_fQ]->SetMaximum(1.1*ymax);
hrelE[ic_fQ]->SetMinimum(0.);
hrelE[ic_fQ]->Draw();
hrelE[ic_h5]->Draw("same");

l1->Draw();
}
