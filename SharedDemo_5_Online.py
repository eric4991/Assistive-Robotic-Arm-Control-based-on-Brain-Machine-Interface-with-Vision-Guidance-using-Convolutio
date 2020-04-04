# @ written by Kyung-Hwan Shim
import numpy as np
import torch.nn as nn
import torch
from Model.SMC_shared import *

net = smcShared(3)
ckpt = torch.load('./ckpt/online.t7')
net.load_state_dict(ckpt['net'])
net.cuda()
device = 'cuda'
def onlineMain(eegData):
    eegData=np.asarray(eegData)
    eegData=np.reshape(eegData,(5,5,751))
    eegData=torch.from_numpy(eegData)

    net.eval()
    with torch.no_grad():
        eegData=eegData[np.newaxis,np.newaxis,:,:]
        eegData=eegData.to(device,dtype=torch.float)
        output=net(eegData)
        output=output.data.cpu().numpy()

    eegData=None
    return output