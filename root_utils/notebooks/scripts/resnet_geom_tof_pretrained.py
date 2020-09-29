import torch.nn as nn
import torch.nn.functional as F
from common import *
from resnet_geom import *

class ResNetGeomTofPretrained(nn.Module):
    """
    Note: the network output will be num_classes+5
          The first five output channels are the vertex prediction by the simple_pos
          network (3 coord in cm and 2 logSigma values).
          So to just select the output channels of the full network,
          use out[:,5:]
    """
    def __init__(self, block, num_blocks, simple_net, num_classes=3):
        super(ResNetGeomTofPretrained, self).__init__()

        self.simple_position = simple_net
        self.full = ResNetGeom(block, num_blocks, num_classes)

    def forward(self, x):
        with torch.no_grad():
            simple_pos_out = self.simple_position(x)
            simple_pos_out = simple_pos_out[:,2:7] # discard everything but position
        x_tof_corrected = self.subtract_tof(x, simple_pos_out)
        full_out = self.full(x_tof_corrected)
        return torch.cat([simple_pos_out, full_out], 1)

    def subtract_tof(self, x, simple_pos_out):
        B = x.shape[0] # batch size

        cmplX = x[:,:,:,0]
        cmplY = x[:,:,:,1]
        evQ,evT = torch_XY2QT(cmplX, cmplY) # evT in ns

        pmtXYZ = x[:,:,:,2:5] # pmt xyz position in 10m
        vtxXYZ = simple_pos_out[:,0:3].view([B,1,1,3]) # predicted vertex position in 10m
        dist = torch.sqrt(torch.sum(torch.pow(pmtXYZ - vtxXYZ,2),3)) * 1000.# distance in cm
        tof = dist / (speedOfLight_cm_ns/refractiveIndex_water) - 70. # in ns
        # the 70 ns offset is between the 890 ns peak we get after tof subtraction
        # and the 960. ns we get before that we have hardcoded in torch_QT2XY

        # meanB = (torch.sum(evQ*evT) / torch.sum(evQ)).detach().item()
        # stdB  = torch.sqrt(torch.sum(evQ * (evT-meanB)**2) / torch.sum(evQ)).detach().item()
        # meanB = (torch.sum(torch.where(evQ > 0., evT, evT*0.)) / torch.sum(evQ > 0.)).detach().item()
        # stdB  = torch.sqrt(torch.sum(torch.where(evQ > 0., (evT-meanB)**2, evT*0.)) / torch.sum(evQ>0.)).detach().item()

        evT = torch.where(evQ > 0., evT - tof, evT)

        # meanA = (torch.sum(evQ*evT) / torch.sum(evQ)).detach().item()
        # stdA  = torch.sqrt(torch.sum(evQ * (evT-meanA)**2) / torch.sum(evQ)).detach().item()
        # meanA = (torch.sum(torch.where(evQ > 0., evT, evT*0.)) / torch.sum(evQ > 0.)).detach().item()
        # stdA  = torch.sqrt(torch.sum(torch.where(evQ > 0., (evT-meanA)**2, evT*0.)) / torch.sum(evQ>0.)).detach().item()
        # print('time before: %g +- %g, after tof subtraction: %g +- %g' % (meanB, stdB, meanA, stdA))

        cmplX,cmplY = torch_QT2XY(evQ, evT)
        rest = x[:,:,:,2:]
        return torch.cat([torch.stack([cmplX,cmplY],3),rest], 3)


def ResNetGeomTofPretrained18(simple_net, num_classes):
    return ResNetGeomTofPretrained(BasicGeomBlock, [2,2,2,2], simple_net, num_classes)
