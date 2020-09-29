from past.builtins import execfile
import os

from common import *
from resnet_geom import *

def getEnergyPrediction(out):
    out_Eabovethres  = out[:,0:1]
    out_logSigmaESqr = out[:,1:2]
    #print('pred energy', torch.mean(out_Eabovethres), torch.std(out_Eabovethres))
    return out_Eabovethres, out_logSigmaESqr

def getPositionPrediction(out):
    out_pos            = out[:,2:5]
    out_logSigmaPosSqr = out[:,5:6]
    return out_pos, out_logSigmaPosSqr

def getDirectionPrediction(out):
    out_dir            = out[:,6:9]
    out_logSigmaDirSqr = out[:,9:10]
    return out_dir, out_logSigmaDirSqr

def getTrueEnergyAboveThreshold(blob):
    Evis = 0.085*blob.totQ
    out = (blob.energy-Ethres[blob.label] - Evis) / (np.sqrt(500.)*np.sqrt(Evis+0.5))
    #print('true energy', np.mean(out), np.std(out))
    return out

def getNllEnergy(out_Eabovethres, out_logSigmaESqr, true_Eabovethres):
    return 0.5*torch.sum(torch.pow(out_Eabovethres - true_Eabovethres, 2)*torch.exp(-out_logSigmaESqr),1) + 0.5*torch.sum(out_logSigmaESqr,1)

def getNllPosition(out_pos, out_logSigmaPosSqr, true_pos):
    return 0.5*torch.sum(torch.pow(out_pos - true_pos, 2)*torch.exp(-out_logSigmaPosSqr),1) + 0.5*3.*torch.sum(out_logSigmaPosSqr,1)

def getNllDirection(out_dir, out_logSigmaDirSqr, true_dir):
    norm = torch.sqrt(torch.sum(torch.pow(out_dir,2),1))+1e-6 # add 1e-6 to prevent division by 0
    angle = torch.acos(torch.sum(out_dir*true_dir, 1)/norm) # [B]
    return 0.5*torch.pow(angle, 2)*torch.exp(-out_logSigmaDirSqr[:,0]) + 0.5*out_logSigmaDirSqr[:,0] + 0.5*torch.pow(norm-1.,2)/(0.1*0.1)


execfile(os.path.dirname(__file__)+"/forward_releposdir.py")
execfile(os.path.dirname(__file__)+"/inference_releposdir.py")

import sys

if len(sys.argv) < 2:
    print("Usage: %s {reco|train}" % sys.argv[0])
    sys.exit(1)

class BLOB:
    pass
blob=BLOB()
blob.net       = ResNetGeom18(10).cuda() # E,Eres,pos[3],posres,dir[3],dirres
# just the training weights is ~700MiB
blob.optimizer = torch.optim.Adam(blob.net.parameters()) # use Adam optimizer algorithm
blob.prefix    = '20200827-01-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir'
blob.epoch     = 0.
blob.iteration = 0
blob.data      = None # data for training/analysis
blob.label     = None # label for training/analysis

restore_state(blob, 364504)

if sys.argv[1] == "reco":
    execfile(os.path.dirname(__file__)+'/process_events.py')
elif sys.argv[1] == "train":
    execfile(os.path.dirname(__file__)+'/train.py')
else:
    print("Don't know what to do: %s" % sys.argv[1])
    sys.exit(70)
