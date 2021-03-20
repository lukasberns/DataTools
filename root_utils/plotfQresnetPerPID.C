#include "TChain.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TMath.h"
#include "TVector3.h"

#include <cmath>
#include <cassert>
#include <vector>
#include <string>
using namespace std;

const int nc = 2;
const int ic_fQ = 0;
const int ic_h5 = 1;

TH1D *hrelE[nc];
TH1D *hpos[nc];
TH1D *hposL[nc];
TH1D *hposT[nc];
TH1D *hposT2[nc];
TH1D *hdir[nc];

TH2D *hrelEtowall[nc];
TH2D *hpostowall[nc];
TH2D *hdirtowall[nc];

TH2D *hrelEE[nc];
TH2D *hposE[nc];
TH2D *hdirE[nc];

TGraph *pid_ROC[nc];

TH1D *hdwall;
TH1D *htowall;

TCanvas *c1 = NULL;

const char *confst[nc];

void plotfQresnetPerPID(const char *recoversion, const char *recovershort, const char *pname="e-", bool use_pass=true, const char *devnotedir = NULL) {

// const char *pname = "e-";
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

const int label_electron = 0;
const int label_muon = 1;
const int label_gamma = 2;
const int nlabel = 3;

int label_sig = label_electron;
int label_bkg = label_muon;

vector<string> pnames;
if (string("e-mu-") == pname) {
  pnames.push_back("e-");
  pnames.push_back("mu-");
}
else if (string("e-gamma") == pname) {
  pnames.push_back("e-");
  pnames.push_back("gamma");
  label_bkg = label_gamma;
}
else {
  pnames.push_back(pname);
}

TChain *fQ = new TChain("fiTQun");
TChain *h5 = new TChain("h5");
const int bchOffset = 2;
const int NentriesPerFile = 3000;
for (int bch = bchOffset; bch < 100; bch++) {
// for (int bch = bchOffset; bch < 61; bch++) {
  for (unsigned ip = 0; ip < pnames.size(); ip++) {
    const char *thispname = pnames.at(ip).c_str();
    fQ->Add(Form("%s/fiTQun/%s/E0to1000MeV/unif-pos-R371-y521cm/4pi-dir/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d_fiTQun.root", datadir, thispname, thispname, bch));
    h5->Add(Form("%s/reco_%s/%s/IWCDmPMT_4pi_full_tank_%s_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_%d.root", datadir, recoversion, thispname, thispname, bch));
  }
}

TChain *todveto = new TChain("h5");
todveto->Add(Form("%s/od_veto.root", datadir));

TChain *tmichels = new TChain("h5");
tmichels->Add(Form("%s/michels.root", datadir));


int fQ_nevt;
const int fiTQun_maxsubevt = 3;
const int fiTQun_npid = 7;
const int fiTQun_electron = 1;
const int fiTQun_muon     = 2;

float fq1rnll[fiTQun_maxsubevt][fiTQun_npid];
float fq1rmom[fiTQun_maxsubevt][fiTQun_npid];
float fq1rpos[fiTQun_maxsubevt][fiTQun_npid][3];
float fq1rdir[fiTQun_maxsubevt][fiTQun_npid][3];
int fq1rpcflg[fiTQun_maxsubevt][fiTQun_npid];

fQ->SetBranchAddress("nevt", &fQ_nevt);
fQ->SetBranchAddress("fq1rnll", fq1rnll);
fQ->SetBranchAddress("fq1rmom", fq1rmom);
fQ->SetBranchAddress("fq1rpos", fq1rpos);
fQ->SetBranchAddress("fq1rdir", fq1rdir);
fQ->SetBranchAddress("fq1rpcflg", fq1rpcflg);

int true_label;

const char *labelLatex[nlabel] = { "e", "#mu", "#gamma" };
const double mass[nlabel] = { 0.511, 105.7, 0.511*2. };
const double pthres[nlabel] = { 0.57, 118., 0.57 *2 };
double Ethres[nlabel];
for (int il = 0; il < nlabel; il++) {
  Ethres[il] = sqrt(pow(mass[il],2) + pow(pthres[il],2));
}

float true_Eabovethres;
float pred_Eabovethres_perpid[nlabel];
float pred_Eabovethres;

float true_position[3];
float pred_position_perpid[3*nlabel];
float *pred_position;

float true_direction[3];
float pred_direction_perpid[3*nlabel];
float *pred_direction;

bool has_pred_pid = false;
float pred_pid_softmax[nlabel];

h5->SetBranchAddress("true_label", &true_label);
h5->SetBranchAddress("true_Eabovethres", &true_Eabovethres);
h5->SetBranchAddress("pred_Eabovethres", &pred_Eabovethres_perpid);
h5->SetBranchAddress("true_positions", true_position);
h5->SetBranchAddress("pred_position" , pred_position_perpid);
//h5->SetBranchAddress("pred_simple_position" , pred_position);
h5->SetBranchAddress("true_directions", true_direction);
h5->SetBranchAddress("pred_direction" , pred_direction_perpid);

if (h5->GetBranch("pred_pid_softmax") != NULL) {
  has_pred_pid = true;
  h5->SetBranchAddress("pred_pid_softmax", pred_pid_softmax);
}

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

  hposT2[ic] = new TH1D(Form("hposT2_%d", ic), ";Transverse distance square to true vertex [cm^{2}];Events", 200, 0., 400.);
  hposT2[ic]->SetLineColor(colors[ic]);
  
  hdir[ic] = new TH1D(Form("hdir_%d", ic), ";Angle to true direction [deg];Events", 200, 0., 45.);
  hdir[ic]->SetLineColor(colors[ic]);


  hrelEtowall[ic] = new TH2D(Form("hrelEtowall_%d", ic), ";Distance to wall along track [cm];|pred - true|/true energy", 50, 0., 520., 200, 0., 0.5);
  hpostowall[ic] = new TH2D(Form("hpostowall_%d", ic), ";Distance to wall along track [cm];Distance to true vertex [cm]", 50, 0., 520., 200, 0., 200.);
  hdirtowall[ic] = new TH2D(Form("hdirtowall_%d", ic), ";Distance to wall along track [cm];Angle to true direction [deg]", 50, 0., 520., 200, 0., 45.);

  hrelEE[ic] = new TH2D(Form("hrelEE_%d", ic), ";True energy above threshold [MeV];|pred - true|/true energy",     50, 0., 1000., 200, 0., 0.5);
  hposE[ic] = new TH2D(Form("hposE_%d", ic),   ";True energy above threshold [MeV];Distance to true vertex [cm]",  50, 0., 1000., 200, 0., 200.);
  hdirE[ic] = new TH2D(Form("hdirE_%d", ic),   ";True energy above threshold [MeV];Angle to true direction [deg]", 50, 0., 1000., 200, 0., 45.);
}



Long_t fQ_Nentries = fQ->GetEntries();
Long64_t *h5TreeOffset = h5->GetTreeOffset();

h5->GetEntry(0);
assert(h5->GetTree()->GetEntries() == NentriesPerFile);

Long_t pid_Nentries = 0;
double *pid_llr[nc];
int *pid_true = new int[fQ_Nentries];
for (int ic = 0; ic < nc; ic++) {
  pid_llr[ic] = new double[fQ_Nentries];
}

for (Long_t fQev = 0; fQev < fQ_Nentries; fQev++) {
  fQ->GetEntry(fQev);
  h5->GetEntry(h5TreeOffset[fQ->GetTreeNumber()] + fQ_nevt-1);
  
  int globalId = (bchOffset + fQ->GetTreeNumber())*NentriesPerFile + fQ_nevt-1;
  todveto->GetEntry(globalId);
  tmichels->GetEntry(globalId);
  
  pred_Eabovethres = pred_Eabovethres_perpid[true_label];
  pred_position    = pred_position_perpid + 3*true_label;
  pred_direction   = pred_direction_perpid + 3*true_label;

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

  if (passCut != use_pass) { continue; }
  
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

  hposT2[ic_fQ]->Fill((fqpos - trpos - ((fqpos - trpos) * trdir)*trdir).Mag2());
  hposT2[ic_h5]->Fill((h5pos - trpos - ((h5pos - trpos) * trdir)*trdir).Mag2());

  
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


  hrelEE[ic_fQ]->Fill(true_Eabovethres, fabs(fq1rEabovethres - true_Eabovethres)/true_Eabovethres);
  hrelEE[ic_h5]->Fill(true_Eabovethres, fabs(pred_Eabovethres - true_Eabovethres)/true_Eabovethres);

  hposE[ic_fQ]->Fill(true_Eabovethres, (fqpos - trpos).Mag());
  hposE[ic_h5]->Fill(true_Eabovethres, (h5pos - trpos).Mag());

  hdirE[ic_fQ]->Fill(true_Eabovethres, trdir.Angle(fqdir)/TMath::Pi()*180.);
  hdirE[ic_h5]->Fill(true_Eabovethres, trdir.Angle(h5dir)/TMath::Pi()*180.);

  if (has_pred_pid) {
    pid_llr[ic_fQ][pid_Nentries] = fq1rnll[0][fiTQun_electron] - fq1rnll[0][fiTQun_muon]; // log L(mu)/L(e)
    pid_llr[ic_h5][pid_Nentries] = log(pred_pid_softmax[label_bkg] / pred_pid_softmax[label_sig]); // log L(bkg)/L(sig)
    pid_true[pid_Nentries] = true_label;
    pid_Nentries++;
  }
}

if (has_pred_pid) {
  Long_t pid_Nentries_sig = 0;
  Long_t pid_Nentries_bkg = 0;
  for (Long_t ev = 0; ev < pid_Nentries; ev++) {
    if (pid_true[ev] == label_sig) {
      pid_Nentries_sig++;
    }
    else if (pid_true[ev] == label_bkg) {
      pid_Nentries_bkg++;
    }
  }
  printf("pid_Nentries_sig = %lld, pid_Nentries_bkg = %lld\n", pid_Nentries_sig, pid_Nentries_bkg);

  Long_t *pid_llr_order[nc];
  double *pid_roc_sigeff[nc];
  double *pid_roc_bkgeff[nc];
  double *pid_roc_bkgrej[nc];
  for (int ic = 0; ic < nc; ic++) {
    pid_llr_order[ic] = new Long_t[pid_Nentries];
    TMath::Sort(pid_Nentries, pid_llr[ic], pid_llr_order[ic], false);
    delete [] pid_llr[ic];

    pid_roc_sigeff[ic] = new double[pid_Nentries];
    pid_roc_bkgeff[ic] = new double[pid_Nentries];
    pid_roc_bkgrej[ic] = new double[pid_Nentries];

    Long_t isb = 0; // sig or bkg
    for (Long_t io = 0; io < pid_Nentries; io++) {
      Long_t ev = pid_llr_order[ic][io];
      if (isb == 0) {
        pid_roc_sigeff[ic][isb] = 0.;
        pid_roc_bkgeff[ic][isb] = 0.;
        pid_roc_bkgrej[ic][isb] = 0.;
      }
      else {
        pid_roc_sigeff[ic][isb] = pid_roc_sigeff[ic][isb-1];
        pid_roc_bkgeff[ic][isb] = pid_roc_bkgeff[ic][isb-1];
        pid_roc_bkgrej[ic][isb] = pid_roc_bkgrej[ic][isb-1];
      }

      if (pid_true[ev] == label_sig) {
        pid_roc_sigeff[ic][isb] += 1./double(pid_Nentries_sig);
        isb++;
      }
      else if (pid_true[ev] == label_bkg) {
        pid_roc_bkgeff[ic][isb] += 1./double(pid_Nentries_bkg);
        pid_roc_bkgrej[ic][isb]  = 1./pid_roc_bkgeff[ic][isb];
        isb++;
      }
    }
    Long_t sigbkg_Nentries = isb;
    printf("sigbkg_Nentries = %lld\n", sigbkg_Nentries);

    for (isb = 0; isb < sigbkg_Nentries; isb++) {
      if (pid_roc_bkgeff[ic][isb] > 0.) {
        break;
      }
    }
    Long_t zerobkg_Nentries = isb;
    printf("zerobkg_Nentries = %lld\n", zerobkg_Nentries);

    if (pid_Nentries_sig > 0 && pid_Nentries_bkg > 0) {
      pid_ROC[ic] = new TGraph(sigbkg_Nentries-zerobkg_Nentries, pid_roc_sigeff[ic]+zerobkg_Nentries, pid_roc_bkgrej[ic]+zerobkg_Nentries);
    }
    else {
      printf("Cannot make pid ROC curves since either sig or bkg have no entries\n");
    }

    delete [] pid_roc_sigeff[ic];
    delete [] pid_roc_bkgeff[ic];
    delete [] pid_roc_bkgrej[ic];
    delete [] pid_llr_order[ic];
  }

  delete [] pid_true;
}

c1 = new TCanvas;

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

if (devnotedir != NULL) {
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_relE.pdf", devnotedir, recoversion, pname, use_pass));

l1 = new TLegend(0.5, 0.65, 0.9, 0.85);
l1->SetBorderSize(0);

///////////// POSITION /////////////
ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hpos[ic], confst[ic], "L");
  if (hpos[ic]->GetMaximum() > ymax) {
    ymax = hpos[ic]->GetMaximum();
  }
}
hpos[ic_fQ]->SetMaximum(1.1*ymax);
hpos[ic_fQ]->SetMinimum(0.);
hpos[ic_fQ]->Draw();
hpos[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_pos.pdf", devnotedir, recoversion, pname, use_pass));

l1 = new TLegend(0.6, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);

ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hposL[ic], confst[ic], "L");
  if (hposL[ic]->GetMaximum() > ymax) {
    ymax = hposL[ic]->GetMaximum();
  }
}
hposL[ic_fQ]->SetMaximum(1.1*ymax);
hposL[ic_fQ]->SetMinimum(0.);
hposL[ic_fQ]->Draw();
hposL[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_posL.pdf", devnotedir, recoversion, pname, use_pass));

l1 = new TLegend(0.5, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);

ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hposT[ic], confst[ic], "L");
  if (hposT[ic]->GetMaximum() > ymax) {
    ymax = hposT[ic]->GetMaximum();
  }
}
hposT[ic_fQ]->SetMaximum(1.1*ymax);
hposT[ic_fQ]->SetMinimum(0.);
hposT[ic_fQ]->Draw();
hposT[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_posT.pdf", devnotedir, recoversion, pname, use_pass));

l1 = new TLegend(0.5, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);

ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hposT2[ic], confst[ic], "L");
  if (hposT[ic]->GetMaximum() > ymax) {
    ymax = hposT2[ic]->GetMaximum();
  }
}
hposT2[ic_fQ]->SetMaximum(1.1*ymax);
hposT2[ic_fQ]->SetMinimum(0.);
hposT2[ic_fQ]->Draw();
hposT2[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_posT2.pdf", devnotedir, recoversion, pname, use_pass));

l1 = new TLegend(0.4, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);

//////////// DIRECTION //////////////
ymax = 1.;
for (int ic = 0; ic < nc; ic++) {
  l1->AddEntry(hdir[ic], confst[ic], "L");
  if (hdir[ic]->GetMaximum() > ymax) {
    ymax = hdir[ic]->GetMaximum();
  }
}
hdir[ic_fQ]->SetMaximum(1.1*ymax);
hdir[ic_fQ]->SetMinimum(0.);
hdir[ic_fQ]->Draw();
hdir[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_dir.pdf", devnotedir, recoversion, pname, use_pass));

//////////// PID ROC curve /////////////
if (pid_ROC[0] != NULL) {
  c1->SetLogy();
  pid_ROC[ic_fQ]->SetLineColor(kAzure-2);
  pid_ROC[ic_fQ]->SetLineWidth(2);
  pid_ROC[ic_fQ]->SetMinimum(1.);
  pid_ROC[ic_fQ]->GetXaxis()->SetRangeUser(0.2, 1.05);
  pid_ROC[ic_fQ]->SetTitle("");
  pid_ROC[ic_fQ]->GetXaxis()->SetTitle(Form("%s signal efficiency", labelLatex[label_sig]);
  pid_ROC[ic_fQ]->GetYaxis()->SetTitle(Form("%s background rejection", labelLatex[label_bkg]);
  pid_ROC[ic_fQ]->Draw("AL");
  
  pid_ROC[ic_h5]->SetLineColor(kRed-4);
  pid_ROC[ic_h5]->SetLineWidth(2);
  pid_ROC[ic_h5]->Draw("L");
  
  l1 = new TLegend(0.17, 0.25, 0.65, 0.45);
  l1->SetBorderSize(0);
  for (int ic = 0; ic < nc; ic++) {
    l1->AddEntry(pid_ROC[ic], confst[ic], "L");
  }
  l1->Draw();
  c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_PID.pdf", devnotedir, recoversion, pname, use_pass));
  c1->SetLogy(0);
}

/////////// relE as function of towall ///////////
TH1D *hrelEtowall_1sig[nc];
for (int ic = 0; ic < nc; ic++) {
  hrelEtowall_1sig[ic] = (TH1D *)hrelEtowall[ic]->ProjectionX(Form("hrelEtowall_1sig_%d", ic));
  hrelEtowall_1sig[ic]->GetYaxis()->SetTitle("Relative energy resolution");
  for (int ix = 0; ix < hrelEtowall[ic]->GetNbinsX(); ix++) {
    double cumsum = 0.;
    int iy;
    for (iy = 0; iy < hrelEtowall[ic]->GetNbinsY(); iy++) {
      cumsum += hrelEtowall[ic]->GetBinContent(ix+1,iy+1);
      if (cumsum > 0.683*hrelEtowall_1sig[ic]->GetBinContent(ix+1)) {
        break;
      }
    }
    hrelEtowall_1sig[ic]->SetBinContent(ix+1, hrelEtowall[ic]->GetYaxis()->GetBinUpEdge(iy+1));
  }
}

c1->cd();
l1 = new TLegend(0.5, 0.65, 0.9, 0.85);
l1->SetBorderSize(0);
ymax = 0.1;
for (int ic = 0; ic < nc; ic++) {
  hrelEtowall_1sig[ic]->SetLineColor(hrelE[ic]->GetLineColor());
  l1->AddEntry(hrelEtowall_1sig[ic], confst[ic], "L");
  if (hrelEtowall_1sig[ic]->GetMaximum() > ymax) {
    ymax = hrelEtowall_1sig[ic]->GetMaximum();
  }
}
ymax = 0.5;
hrelEtowall_1sig[ic_fQ]->SetMaximum(0.3);
hrelEtowall_1sig[ic_fQ]->SetMinimum(0.);
hrelEtowall_1sig[ic_fQ]->Draw();
hrelEtowall_1sig[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_relE_towall.pdf", devnotedir, recoversion, pname, use_pass));

/////////// pos as function of towall ////////////
TH1D *hpostowall_1sig[nc];
for (int ic = 0; ic < nc; ic++) {
  hpostowall_1sig[ic] = (TH1D *)hpostowall[ic]->ProjectionX(Form("hpostowall_1sig_%d", ic));
  hpostowall_1sig[ic]->GetYaxis()->SetTitle(hpostowall[ic]->GetYaxis()->GetTitle());
  for (int ix = 0; ix < hpostowall[ic]->GetNbinsX(); ix++) {
    double cumsum = 0.;
    int iy;
    for (iy = 0; iy < hpostowall[ic]->GetNbinsY(); iy++) {
      cumsum += hpostowall[ic]->GetBinContent(ix+1,iy+1);
      if (cumsum > 0.683*hpostowall_1sig[ic]->GetBinContent(ix+1)) {
        break;
      }
    }
    hpostowall_1sig[ic]->SetBinContent(ix+1, hpostowall[ic]->GetYaxis()->GetBinUpEdge(iy+1));
  }
}

c1->cd();
l1 = new TLegend(0.45, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);
ymax = 0.1;
for (int ic = 0; ic < nc; ic++) {
  hpostowall_1sig[ic]->SetLineColor(hpos[ic]->GetLineColor());
  l1->AddEntry(hpostowall_1sig[ic], confst[ic], "L");
  if (hpostowall_1sig[ic]->GetMaximum() > ymax) {
    ymax = hpostowall_1sig[ic]->GetMaximum();
  }
}
ymax = 30.;
hpostowall_1sig[ic_fQ]->SetMaximum(1.1*ymax);
hpostowall_1sig[ic_fQ]->SetMinimum(0.);
hpostowall_1sig[ic_fQ]->Draw();
hpostowall_1sig[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_pos_towall.pdf", devnotedir, recoversion, pname, use_pass));

////////////// dir as function of towall /////////////
TH1D *hdirtowall_1sig[nc];
for (int ic = 0; ic < nc; ic++) {
  hdirtowall_1sig[ic] = (TH1D *)hdirtowall[ic]->ProjectionX(Form("hdirtowall_1sig_%d", ic));
  hdirtowall_1sig[ic]->GetYaxis()->SetTitle(hdirtowall[ic]->GetYaxis()->GetTitle());
  for (int ix = 0; ix < hdirtowall[ic]->GetNbinsX(); ix++) {
    double cumsum = 0.;
    int iy;
    for (iy = 0; iy < hdirtowall[ic]->GetNbinsY(); iy++) {
      cumsum += hdirtowall[ic]->GetBinContent(ix+1,iy+1);
      if (cumsum > 0.683*hdirtowall_1sig[ic]->GetBinContent(ix+1)) {
        break;
      }
    }
    hdirtowall_1sig[ic]->SetBinContent(ix+1, hdirtowall[ic]->GetYaxis()->GetBinUpEdge(iy+1));
  }
}

c1->cd();
l1 = new TLegend(0.45, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);
ymax = 0.1;
for (int ic = 0; ic < nc; ic++) {
  hdirtowall_1sig[ic]->SetLineColor(hdir[ic]->GetLineColor());
  l1->AddEntry(hdirtowall_1sig[ic], confst[ic], "L");
  if (hdirtowall_1sig[ic]->GetMaximum() > ymax) {
    ymax = hdirtowall_1sig[ic]->GetMaximum();
  }
}
ymax = 20.;
hdirtowall_1sig[ic_fQ]->SetMaximum(1.1*ymax);
hdirtowall_1sig[ic_fQ]->SetMinimum(0.);
hdirtowall_1sig[ic_fQ]->Draw();
hdirtowall_1sig[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_dir_towall.pdf", devnotedir, recoversion, pname, use_pass));


////////////// relE as function of energy //////////////
TH1D *hrelEE_1sig[nc];
for (int ic = 0; ic < nc; ic++) {
  hrelEE_1sig[ic] = (TH1D *)hrelEE[ic]->ProjectionX(Form("hrelEE_1sig_%d", ic));
  hrelEE_1sig[ic]->GetYaxis()->SetTitle("Relative energy resolution");
  for (int ix = 0; ix < hrelEE[ic]->GetNbinsX(); ix++) {
    double cumsum = 0.;
    int iy;
    for (iy = 0; iy < hrelEE[ic]->GetNbinsY(); iy++) {
      cumsum += hrelEE[ic]->GetBinContent(ix+1,iy+1);
      if (cumsum > 0.683*hrelEE_1sig[ic]->GetBinContent(ix+1)) {
        break;
      }
    }
    hrelEE_1sig[ic]->SetBinContent(ix+1, hrelEE[ic]->GetYaxis()->GetBinUpEdge(iy+1));
  }
}

c1->cd();
l1 = new TLegend(0.45, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);
ymax = 0.1;
for (int ic = 0; ic < nc; ic++) {
  hrelEE_1sig[ic]->SetLineColor(hrelE[ic]->GetLineColor());
  l1->AddEntry(hrelEE_1sig[ic], confst[ic], "L");
  if (hrelEE_1sig[ic]->GetMaximum() > ymax) {
    ymax = hrelEE_1sig[ic]->GetMaximum();
  }
}
ymax=0.2/1.1;
hrelEE_1sig[ic_fQ]->SetMaximum(1.1*ymax);
hrelEE_1sig[ic_fQ]->SetMinimum(0.);
hrelEE_1sig[ic_fQ]->Draw();
hrelEE_1sig[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_relE_E.pdf", devnotedir, recoversion, pname, use_pass));

//////////////// pos as function of energy ///////////
TH1D *hposE_1sig[nc];
for (int ic = 0; ic < nc; ic++) {
  hposE_1sig[ic] = (TH1D *)hposE[ic]->ProjectionX(Form("hposE_1sig_%d", ic));
  hposE_1sig[ic]->GetYaxis()->SetTitle(hposE[ic]->GetYaxis()->GetTitle());
  for (int ix = 0; ix < hposE[ic]->GetNbinsX(); ix++) {
    double cumsum = 0.;
    int iy;
    for (iy = 0; iy < hposE[ic]->GetNbinsY(); iy++) {
      cumsum += hposE[ic]->GetBinContent(ix+1,iy+1);
      if (cumsum > 0.683*hposE_1sig[ic]->GetBinContent(ix+1)) {
        break;
      }
    }
    hposE_1sig[ic]->SetBinContent(ix+1, hposE[ic]->GetYaxis()->GetBinUpEdge(iy+1));
  }
}

c1->cd();
l1 = new TLegend(0.45, 0.65, 0.95, 0.85);
l1->SetBorderSize(0);
ymax = 0.1;
for (int ic = 0; ic < nc; ic++) {
  hposE_1sig[ic]->SetLineColor(hpos[ic]->GetLineColor());
  l1->AddEntry(hposE_1sig[ic], confst[ic], "L");
  if (hposE_1sig[ic]->GetMaximum() > ymax) {
    ymax = hposE_1sig[ic]->GetMaximum();
  }
}
ymax = 30.;
hposE_1sig[ic_fQ]->SetMaximum(1.1*ymax);
hposE_1sig[ic_fQ]->SetMinimum(0.);
hposE_1sig[ic_fQ]->Draw();
hposE_1sig[ic_h5]->Draw("same");
l1->Draw();
c1->SaveAs(Form("%s/fqresnet_%s_%s_pass%d_pos_E.pdf", devnotedir, recoversion, pname, use_pass));
}
}
