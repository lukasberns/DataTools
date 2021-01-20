import torch.nn as nn
import torch.nn.functional as F
from common import *

# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

class BasicGeomBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicGeomBlock, self).__init__()
        
        self.pad = 1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=2*self.pad+1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=2*self.pad+1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(geometricPad(x,self.pad))))
        out = self.bn2(self.conv2(geometricPad(out,self.pad)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetGeom(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3, overall_in_planes=8, overall_ch=64, initial_stride=[2,2], use_layer0=False):
        # overall_in_planes: 8 for Q+T+3PMTpos+3PMTdir
        # for mPMT_as_layers: 44 for 19Q+19T+3PMTpos+3PMTdir

        # for mPMT_as_layers use initial_stride=(1,1)
        # and use_layer0 = True
        super(ResNetGeom, self).__init__()
        self.pad = 1
        
        # (1,126,126)
        # ( , 29, 29) for mPMT_as_layers
        self.conv1 = nn.Conv2d(overall_in_planes, overall_ch, kernel_size=2*self.pad+1,
                               stride=initial_stride[0], padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(overall_ch)
        # outsize = floor[(126+2*self.pad-3)/stride + 1]
        # so for stride=1 we have 126,
        #              =2          63 = floor(63.5)
        # the padding is done by geometricPad
        # thus (64,63,63)
        #      (64,29,29) for mPMT_as_layers with initial_stride=(1,)
        self.conv2 = nn.Conv2d(overall_ch, overall_ch, kernel_size=2*self.pad+1,
                               stride=initial_stride[1], padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(overall_ch)
        # outsize = floor[(63+2*self.pad-3)/stride + 1]
        # so for stride=1 we have 63,
        #              =2         32
        # thus (64,32,32)
        #      (64,29,29) for mPMT_as_layers with initial_stride=(1,1)

        self.in_planes = overall_ch # match with output of bn2
        self.layer0 = None
        if use_layer0:
            self.layer1 = block(overall_ch,overall_ch)
        self.layer1 = self._make_layer(block,   overall_ch, num_blocks[0], stride=(2 if use_layer0 else 1))
        #      (64,32,32)
        #      (64,15,15) for mPMT_as_layers with use_layer0
        self.layer2 = self._make_layer(block,   overall_ch, num_blocks[1], stride=2)
        #      (64,16,16)
        #      (64, 8, 8) for mPMT_as_layers
        self.layer3 = self._make_layer(block,   overall_ch, num_blocks[2], stride=2)
        #      (64, 8, 8)
        #      (64, 4, 4) for mPMT_as_layers
        self.layer4 = self._make_layer(block, 2*overall_ch, num_blocks[3], stride=2)
        #      (128,2, 2)
        flattening_kernel_size = 4
        if use_layer0:
            flattening_kernel_size = 2
        self.convbn5 = nn.Sequential(
            nn.Conv2d(2*overall_ch,2*overall_ch,kernel_size=flattening_kernel_size,stride=1,padding=0),
            nn.BatchNorm2d(2*overall_ch)
        )
        #      (128,1, 1)
        # transform to 128, then
        self.linear1 = nn.Linear(int(2*overall_ch*block.expansion),   int(2*overall_ch*block.expansion/2))
        self.linear2 = nn.Linear(int(2*overall_ch*block.expansion/2), int(2*overall_ch*block.expansion/4))
        self.linear3 = nn.Linear(int(2*overall_ch*block.expansion/4), int(2*overall_ch*block.expansion/8))
        self.linear4 = nn.Linear(int(2*overall_ch*block.expansion/8), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(geometricPad(x,self.pad))))
        out = F.relu(self.bn2(self.conv2(geometricPad(out,self.pad))))
        if self.layer0 is not None:
            out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.convbn5(out))
        #out = F.avg_pool2d(out, out.shape[2:4])
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out

#    # I'm not sure why I need to implement cpu/cuda myself, train works somehow
#    
#    def cpu(self):
#        super(GeomCNN, self).cpu()
#        for module in self._feature:
#            module.cpu()
#        for module in self._classifier:
#            module.cpu()
#        return self
#    
#    def cuda(self):
#        super(GeomCNN, self).cuda()
#        for module in self._feature:
#            module.cuda()
#        for module in self._classifier:
#            module.cuda()
#        return self

def ResNetGeom18(num_classes):
    return ResNetGeom(BasicGeomBlock, [2, 2, 2, 2], num_classes)
