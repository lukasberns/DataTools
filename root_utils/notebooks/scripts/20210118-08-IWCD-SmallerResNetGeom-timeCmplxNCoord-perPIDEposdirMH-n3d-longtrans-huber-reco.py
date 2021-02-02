from past.builtins import execfile
import os

from common import *
from resnet_mh_geom import *

npid = 3

def getPIDPrediction(out):
    out_onehot = out[:,0:npid]
    return out_onehot

def getEnergyPrediction(out):
    out_Eabovethres  = out[:,(npid*1):(npid*2)]
    out_logSigmaESqr = out[:,(npid*2):(npid*3)]
    #print('pred energy', torch.mean(out_Eabovethres), torch.std(out_Eabovethres))
    return out_Eabovethres, out_logSigmaESqr

def getPositionPrediction(out):
    out_pos            = out[:,(npid*3):(npid*6)]
    out_logSigmaPosSqr = out[:,(npid*6):(npid*8)]
    return out_pos, out_logSigmaPosSqr

def getDirectionPrediction(out):
    out_dir            = out[:,(npid* 8):(npid*11)]
    out_logSigmaDirSqr = out[:,(npid*11):(npid*12)]
    return out_dir, out_logSigmaDirSqr

def getTruePID(blob):
    return blob.label

def getTrueEnergyAboveThreshold(blob):
    out = blob.energy-Ethres[blob.label]
    #print('true energy', np.mean(out), np.std(out))
    return out

def getNllPid(out_pid, true_pid):
    return torch.nn.functional.cross_entropy(out_pid, true_pid)

def getNllEnergy(out_Eabovethres, out_logSigmaESqr, true_Eabovethres, true_pid):
    out_Eabovethres  = out_Eabovethres .gather(1, true_pid.unsqueeze(1))
    out_logSigmaESqr = out_logSigmaESqr.gather(1, true_pid.unsqueeze(1))
    dE = (out_Eabovethres - true_Eabovethres) * torch.exp(-0.5*out_logSigmaESqr)
    return torch.nn.functional.smooth_l1_loss(dE, 0.*dE) + 0.5*torch.sum(out_logSigmaESqr,1)

def getNllPosition(out_pos, out_logSigmaPosSqr, true_pos, true_dir, true_pid):
    out_pos = out_pos           .gather(1, torch.stack([3*true_pid+0,3*true_pid+1,3*true_pid+2],1))
    lsprL   = out_logSigmaPosSqr.gather(1, (2*true_pid+0).unsqueeze(1))
    lsprT   = out_logSigmaPosSqr.gather(1, (2*true_pid+1).unsqueeze(1))

    d = out_pos - true_pos
    dL = torch.unsqueeze(torch.sum(d * true_dir, 1), 1)
    dT = d - dL * true_dir

    return torch.nn.functional.smooth_l1_loss(dT*torch.exp(-0.5*lsprT),0.*dT) + \
           torch.nn.functional.smooth_l1_loss(dL*torch.exp(-0.5*lsprL),0.*dL) + \
           0.5*(lsprL + lsprT)

def getNllDirection(out_dir, out_logSigmaDirSqr, true_dir, true_pid):
    out_dir = out_dir.gather(1, torch.stack([3*true_pid+0,3*true_pid+1,3*true_pid+2],1))
    out_logSigmaDirSqr = out_logSigmaDirSqr.gather(1, true_pid.unsqueeze(1))

    dn = (out_dir - true_dir) * torch.exp(-0.5*out_logSigmaDirSqr)
    return torch.nn.functional.smooth_l1_loss(dn, 0.*dn) + 0.5*torch.sum(out_logSigmaDirSqr,1)


execfile(os.path.dirname(__file__)+"/forward_perpidreleposdir_longtrans.py")
execfile(os.path.dirname(__file__)+"/inference_pidreleposdir.py")

import sys

if len(sys.argv) < 2:
    print("Usage: %s {reco|train}" % sys.argv[0])
    sys.exit(1)

class BLOB:
    pass
blob=BLOB()
blob.net       = ResNetMHGeom18(npid*np.array([1, 1+1, 3+2, 3+1])).cuda() # {pid[3]},and for each pid {E,Eres},{pos[3],posres[2]},{dir[3],dirres}
# just the training weights is ~700MiB
blob.optimizer = torch.optim.Adam(blob.net.parameters()) # use Adam optimizer algorithm
blob.prefix    = '20210118-08-IWCD-SmallResNetGeom-timeCmplxNCoord-perPIDEposdirMH-01-n3d-longtrans-huber'
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
