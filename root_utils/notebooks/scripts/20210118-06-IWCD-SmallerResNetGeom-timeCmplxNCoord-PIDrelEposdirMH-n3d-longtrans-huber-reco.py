from past.builtins import execfile
import os

from common import *
from resnet_mh_geom import *

def getPIDPrediction(out):
    out_onehot = out[:,0:3]
    return out_onehot

def getEnergyPrediction(out):
    out_Eabovethres  = out[:,3:4]
    out_logSigmaESqr = out[:,4:5]
    #print('pred energy', torch.mean(out_Eabovethres), torch.std(out_Eabovethres))
    return out_Eabovethres, out_logSigmaESqr

def getPositionPrediction(out):
    out_pos            = out[:,5:8]
    out_logSigmaPosSqr = out[:,8:10]
    return out_pos, out_logSigmaPosSqr

def getDirectionPrediction(out):
    out_dir            = out[:,10:13]
    out_logSigmaDirSqr = out[:,13:14]
    return out_dir, out_logSigmaDirSqr

def getTruePID(blob):
    return blob.label

def getTrueEnergyAboveThreshold(blob):
    Evis = 0.085*blob.totQ
    out = (blob.energy-Ethres[blob.label] - Evis) / (np.sqrt(500.)*np.sqrt(Evis+0.5))
    #print('true energy', np.mean(out), np.std(out))
    return out

def getNllPid(out_pid, true_pid):
    return torch.nn.functional.cross_entropy(out_pid, true_pid)

def getNllEnergy(out_Eabovethres, out_logSigmaESqr, true_Eabovethres):
    dE = (out_Eabovethres - true_Eabovethres) * torch.exp(-0.5*out_logSigmaESqr)
    return torch.nn.functional.smooth_l1_loss(dE, 0.*dE) + 0.5*torch.sum(out_logSigmaESqr,1)

def getNllPosition(out_pos, out_logSigmaPosSqr, true_pos, true_dir):
    d = out_pos - true_pos
    dL = torch.unsqueeze(torch.sum(d * true_dir, 1), 1)
    dT = d - dL * true_dir

    lsprL = out_logSigmaPosSqr[:,0:1]
    lsprT = out_logSigmaPosSqr[:,1:2]

    return torch.nn.functional.smooth_l1_loss(dT*torch.exp(-0.5*lsprT),0.*dT) + \
           torch.nn.functional.smooth_l1_loss(dL*torch.exp(-0.5*lsprL),0.*dL) + \
           0.5*torch.sum(out_logSigmaPosSqr,1)

def getNllDirection(out_dir, out_logSigmaDirSqr, true_dir):
    dn = (out_dir - true_dir) * torch.exp(-0.5*out_logSigmaDirSqr)
    return torch.nn.functional.smooth_l1_loss(dn, 0.*dn) + 0.5*torch.sum(out_logSigmaDirSqr,1)



execfile(os.path.dirname(__file__)+"/forward_pidreleposdir_longtrans.py")
execfile(os.path.dirname(__file__)+"/inference_pidreleposdir.py")

import sys

if len(sys.argv) < 2:
    print("Usage: %s {reco|train}" % sys.argv[0])
    sys.exit(1)

class BLOB:
    pass
blob=BLOB()
blob.net       = ResNetMHGeom18([3, 1+1, 3+2, 3+1]).cuda() # {pid[3]},{E,Eres},{pos[3],posres[2]},{dir[3],dirres}
# just the training weights is ~700MiB
blob.optimizer = torch.optim.Adam(blob.net.parameters()) # use Adam optimizer algorithm
blob.prefix    = '20210118-06-IWCD-SmallResNetGeom-timeCmplxNCoord-PIDrelEposdirMH-01-relEposdir-n3d-longtrans-huber'
blob.epoch     = 0.
blob.iteration = 0
blob.data      = None # data for training/analysis
blob.label     = None # label for training/analysis



if sys.argv[1] == "reco":
    blob.prefix += '-20210124-231712'
    #restore_state(blob,  909318) # epoch 12
    restore_state(blob, 2166326) # epoch 28.5
    blob.prefix += '-%d' % (blob.iteration)
    execfile(os.path.dirname(__file__)+'/process_events_pidreleposdir.py')
elif sys.argv[1] == "reco-muon":
    blob.prefix += '-20210124-231712'
    #restore_state(blob, 909318) # epoch 12
    restore_state(blob, 2166326) # epoch 28.5
    blob.prefix += '-%d' % (blob.iteration)
    pname = 'mu-'
    execfile(os.path.dirname(__file__)+'/process_events_pidreleposdir.py')
elif sys.argv[1] == "reco-gamma":
    blob.prefix += '-20210124-231712'
    #restore_state(blob, 909318) # epoch 12
    restore_state(blob, 2166326) # epoch 28.5
    blob.prefix += '-%d' % (blob.iteration)
    pname = 'gamma'
    execfile(os.path.dirname(__file__)+'/process_events_pidreleposdir.py')
elif sys.argv[1] == "reco-electron":
    blob.prefix += '-electron-20200921-043141'
    restore_state(blob, 99939)
    execfile(os.path.dirname(__file__)+'/process_events_pidreleposdir.py')
elif sys.argv[1] == "train":
    from datetime import datetime
    now = datetime.now() # current date and time
    blob.prefix += '-'+now.strftime("%Y%m%d-%H%M%S")
    print(blob.prefix)
    TRAIN_EPOCH = 12.0
    execfile(os.path.dirname(__file__)+'/train.py')
elif sys.argv[1] == "train-long":
    blob.prefix += '-20210124-231712'
    print(blob.prefix)
    restore_state(blob, 909318) # epoch 12
    TRAIN_EPOCH = 40.0
    execfile(os.path.dirname(__file__)+'/train.py')
elif sys.argv[1] == "train-electron":
    from datetime import datetime
    now = datetime.now() # current date and time
    pnameset = "electron"
    blob.prefix += '-electron-'+now.strftime("%Y%m%d-%H%M%S")
    print(blob.prefix)
    TRAIN_EPOCH = 12.0
    execfile(os.path.dirname(__file__)+'/train.py')
else:
    print("Don't know what to do: %s" % sys.argv[1])
    sys.exit(70)
