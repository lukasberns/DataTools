import torch.nn as nn
import torch.nn.functional as F
from common import *
from resnet_geom import *

class ResNetGeomTof(nn.Module):
    """
    Note: the network output will be num_classes+3
          The first five output channels are the vertex prediction by the simple_pos
          network (3 coord in cm and 2 logSigma values).
          So to just select the output channels of the full network,
          use out[:,5:]
    """
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNetGeomTof, self).__init__()

        self.simple_position = ResNetGeom(block, [1,1,1,1], num_classes=5, overall_in_planes=8, overall_ch=8)
        self.full = ResNetGeom(block, num_blocks, num_classes)

    def forward(self, x):
        simple_pos_out = self.simple_position(x)
        x_tof_corrected = self.subtract_tof(x, simple_pos_out)
        full_out = self.full(x_tof_corrected)
        return torch.cat([simple_pos_out, full_out], 1)

    def subtract_tof(self, x, simple_pos_out):
        B = x.shape[0] # batch size

        cmplX = x[:,:,:,0]
        cmplY = x[:,:,:,1]
        evQ,evT = torch_XY2QT(cmplX, cmplY) # evT in ns

        pmtXYZ = x[:,:,:,2:5] # pmt xyz position in m
        vtxXYZ = simple_pos_out[:,0:3].view([B,1,1,3]) # predicted vertex position in m
        dist = torch.sqrt(torch.sum(torch.pow(pmtXYZ - vtxXYZ,2),3)) # distance in m
        tof = dist / (speedOfLight_cm_ns/refractiveIndex_water) # in ns

        cmplX,cmplY = torch_QT2XY(evQ, evT-tof)
        rest = x[:,:,:,2:]
        return torch.cat([torch.stack([cmplX,cmplY],3),rest], 3)


def ResNetGeomTof18(num_classes):
    return ResNetGeomTof(BasicGeomBlock, [2,2,2,2], num_classes)
