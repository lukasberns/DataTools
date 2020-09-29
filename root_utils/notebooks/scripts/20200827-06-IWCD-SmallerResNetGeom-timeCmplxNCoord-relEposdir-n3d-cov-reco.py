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
    out_logSigmaPosSqr = out[:,5:11]
    return out_pos, out_logSigmaPosSqr

def getDirectionPrediction(out):
    out_dir            = out[:,11:14]
    out_logSigmaDirSqr = out[:,14:15]
    return out_dir, out_logSigmaDirSqr

def getTrueEnergyAboveThreshold(blob):
    Evis = 0.085*blob.totQ
    out = (blob.energy-Ethres[blob.label] - Evis) / (np.sqrt(500.)*np.sqrt(Evis+0.5))
    #print('true energy', np.mean(out), np.std(out))
    return out

def getNllEnergy(out_Eabovethres, out_logSigmaESqr, true_Eabovethres):
    return 0.5*torch.sum(torch.pow(out_Eabovethres - true_Eabovethres, 2)*torch.exp(-out_logSigmaESqr),1) + 0.5*torch.sum(out_logSigmaESqr,1)

def getNllPosition(out_pos, out_posres, true_pos):
    d = out_pos - true_pos

    # lower triangular matrix
    L00 = out_posres[:,0]
    L10 = out_posres[:,1]
    L11 = out_posres[:,2]
    L20 = out_posres[:,3]
    L21 = out_posres[:,4]
    L22 = out_posres[:,5]
    # inv(C) = L L^t
    epsilon = 1e-6 ** 2; # a minimum resolution just in case for numerical stability
    # invC = torch.einsum('bij,bkj->bik', L, L) # + epsilon*torch.unsqueeze(torch.eye(3),0)
    logdetC = L00*L11*L22
    logdetC = -torch.log(logdetC*logdetC + epsilon) # need negative sign because LLt is invC

    Ltd = out_pos.new_zeros(out_pos.shape)
    Ltd[:,0] = L00 * d[:,0] + L10 * d[:,1] + L20 * d[:,2]
    Ltd[:,1] =                L11 * d[:,1] + L21 * d[:,2]
    Ltd[:,2] =                               L22 * d[:,2]
    # d invC d = d^T L L^T d = (L^T d)^T (L^T d)

    return 0.5*torch.sum(torch.pow(Ltd,2),1) + 0.5*logdetC

def getNllDirection(out_dir, out_logSigmaDirSqr, true_dir):
    return 0.5*torch.sum(torch.pow(out_dir - true_dir, 2)*torch.exp(-out_logSigmaDirSqr),1) + 0.5*3.*torch.sum(out_logSigmaDirSqr,1)



execfile(os.path.dirname(__file__)+"/forward_releposdir.py")
execfile(os.path.dirname(__file__)+"/inference_releposdir.py")

import sys

if len(sys.argv) < 2:
    print("Usage: %s {reco|train}" % sys.argv[0])
    sys.exit(1)

class BLOB:
    pass
blob=BLOB()
blob.net       = ResNetGeom18(15).cuda() # E,Eres,pos[3],posres[6],dir[3],dirres
# just the training weights is ~700MiB
blob.optimizer = torch.optim.Adam(blob.net.parameters()) # use Adam optimizer algorithm
blob.prefix    = '20200827-06-IWCD-SmallResNetGeom-timeCmplxNCoord-relEposdir-01-relEposdir-n3d-cov'
blob.epoch     = 0.
blob.iteration = 0
blob.data      = None # data for training/analysis
blob.label     = None # label for training/analysis



if sys.argv[1] == "reco":
    blob.prefix += '-20200924-215537'
    restore_state(blob, 909318)
    execfile(os.path.dirname(__file__)+'/process_events.py')
elif sys.argv[1] == "reco-electron":
    blob.prefix += '-electron-20200921-043141'
    restore_state(blob, 99939)
    execfile(os.path.dirname(__file__)+'/process_events.py')
elif sys.argv[1] == "train":
    from datetime import datetime
    now = datetime.now() # current date and time
    blob.prefix += '-'+now.strftime("%Y%m%d-%H%M%S")
    print(blob.prefix)
    TRAIN_EPOCH = 12.0
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
