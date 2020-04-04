import torch
import torch.nn as nn

class smcShared(nn.Module):
    def __init__(self,outputSize):
        self.outputSize=outputSize
        super(smcShared,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv3d(1,36,kernel_size=[1,1,65],padding=0),
            nn.BatchNorm3d(36),
            nn.ELU(True),
        )
        self.conv2=nn.Sequential(
            nn.Conv3d(36,36,kernel_size=[1,1,65],padding=0),
            nn.BatchNorm3d(36),
            nn.ELU(True),
        )
        self.convPool=nn.Sequential(
            nn.Conv3d(36,36,kernel_size=[3,3,6],stride=[1,1,3],padding=0),
            nn.BatchNorm3d(36),
            nn.ELU(True)
        )
        self.conv3=nn.Sequential(
            nn.Conv3d(36,72,kernel_size=[1,1,33],padding=0),
            nn.BatchNorm3d(72),
            nn.ELU(True),
        )
        self.conv4=nn.Sequential(
            nn.Conv3d(72,72,kernel_size=[1,1,33],padding=0),
            nn.BatchNorm3d(72),
            nn.ELU(True),
        )
        self.convPool1=nn.Sequential(
            nn.Conv3d(72,72,kernel_size=[3,3,6],stride=[1,1,3],padding=0),
            nn.BatchNorm3d(72),
            nn.ELU(True),
        )
        self.conv5=nn.Sequential(
            nn.Conv3d(72,144,kernel_size=[1,1,20],padding=0),
            nn.BatchNorm3d(144),
            nn.ELU(True)
        )
        self.conv6=nn.Sequential(
            nn.Conv3d(144,144,kernel_size=[1,1,20],padding=0),
            nn.BatchNorm3d(144),
            nn.ELU(True),
        )
        self.avgpool=nn.AvgPool3d([1,1,3],stride=[1,1,3])
        self.linearOnly=nn.Linear(288,self.outputSize)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.convPool(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out=self.convPool1(out)
        out=self.conv5(out)
        out=self.conv6(out)
        out=self.avgpool(out)
        out=out.view(out.size(0),-1)
        out=self.linearOnly(out)
        out=self.softmax(out)
        return out

def verification():
    x=torch.randn(2,1,5,5,751)
    net=smcShared(5)
    y=net(x)
    print(y)

#verification()