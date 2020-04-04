import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import scipy.io
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os

from Model import *

inputDimIdx=1
subjIdx=0
classIdx=0
# append rest when restIdx is not 0
trainingAmount=40
restIdx=0

device='cuda: 1'
fsIdx=1
modIdx=0

startEpoch=0
trainingEpoch=300
batchSize=32
bestAcc=0

totalTraining=5

fs=['100','250','1000']
mode=['MI','realMove']
classes=[['Forward','Grasp','Twist'],['Left','Right','Forward','Backward','Grasp','Twist'],['Left','Right','Forward','Backward','Up','Down','Grasp','Twist']]
classType=classes[classIdx]
if restIdx==0:
    classNum=len(classType)
else:
    classType=np.append(classType,'Rest')
    classNum=len(classType)
    print(classType)
trials=0
GT=[]

for cIdx,class_ in enumerate(classType):
    if inputDimIdx==0:
        dir_='./2_2DData/'+mode[modIdx]+'/'+fs[fsIdx]+'/'+class_+'/sub'+str(subjIdx+1)
        Data=scipy.io.loadmat(dir_)
        tmpData=np.array(Data[class_+'Data'])
        trials=np.size(tmpData,2)
        tmpTrainData=tmpData[:,:,:trainingAmount]
        tmpTestData=tmpData[:,:,trainingAmount:]
        if cIdx==0:
            trainData=tmpTrainData
            testData=tmpTestData
            trainGT=np.zeros(np.size(tmpTrainData,2))+cIdx
            testGT=np.zeros(np.size(tmpTestData,2))+cIdx
        else:
            trainData=np.append(trainData,tmpTrainData,axis=2)
            testData=np.append(testData,tmpTestData,axis=2)
            trainGT=np.append(trainGT,np.zeros(np.size(tmpTrainData,2))+cIdx)
            testGT=np.append(testGT,np.zeros(np.size(tmpTestData,2))+cIdx)

    else:
        dir_='./3_3DSpatial/'+mode[modIdx]+'/'+fs[fsIdx]+'/'+class_+'/sub'+str(subjIdx+1)
        Data=scipy.io.loadmat(dir_)
        tmpData=np.array(Data[class_+'Data'])
        trials=np.size(tmpData,3)
        tmpTrainData=tmpData[:,:,:,:trainingAmount]
        tmpTestData=tmpData[:,:,:,trainingAmount:]
        if cIdx==0:
            trainData=tmpTrainData
            testData=tmpTestData
            trainGT=np.zeros(np.size(tmpTrainData,3))+cIdx
            testGT=np.zeros(np.size(tmpTestData,3))+cIdx
        else:
            trainData=np.append(trainData,tmpTrainData,axis=3)
            testData=np.append(testData,tmpTestData,axis=3)
            trainGT=np.append(trainGT,np.zeros(np.size(tmpTrainData,3))+cIdx)
            testGT=np.append(testGT,np.zeros(np.size(tmpTestData,3))+cIdx)

if inputDimIdx==0:
    trainData=np.transpose(trainData,(2,1,0))
    testData=np.transpose(testData,(2,1,0))
else:
    trainData=np.transpose(trainData,(3,1,2,0))
    testData=np.transpose(testData,(3,1,2,0))

TrainX=torch.from_numpy(trainData)
TestX=torch.from_numpy(testData)
TrainY=torch.from_numpy(trainGT)
TestY=torch.from_numpy(testGT)

Train=torch.utils.data.TensorDataset(TrainX,TrainY)
Test=torch.utils.data.TensorDataset(TestX,TestY)

trainLoader=torch.utils.data.DataLoader(Train,batch_size=batchSize,shuffle=True)
testLoader=torch.utils.data.DataLoader(Test,batch_size=batchSize,shuffle=True)

def train(epoch):
    print('\n Epoch: %d' %epoch)
    net.train()
    trainLoss=0
    correct=0
    total=0
    for batchIdx,(inputs,targets) in enumerate(trainLoader):
        inputs=inputs[:,np.newaxis,:,:]
        inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.long)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        trainLoss+=loss.item()
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()
def testing(epoch):
    global bestAcc
    net.eval()
    testLoss=0
    correct=0
    total=0
    with torch.no_grad():
        for batchIdx,(inputs,targets) in enumerate(testLoader):
            inputs=inputs[:,np.newaxis,:,:]
            inputs,targets=inputs.to(device,dtype=torch.float),targets.to(device,dtype=torch.long)
            outputs=net(inputs)
            loss=criterion(outputs,targets)
            testLoss+=loss.item()
            _,predicted=outputs.max(1)
            total+=targets.size(0)
            correct+=predicted.eq(targets).sum().item()
    acc=100*correct/total
    print(acc)
    if acc>bestAcc:
        print('Saving...')
        state={
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('ckpt'):
            os.mkdir('ckpt')
        torch.save(state,'./ckpt/ckpt_'+mode[modIdx]+'_hz'+fs[fsIdx]+'_subj'+str(subjIdx)+'.t7')
        bestAcc=acc
net='hi'

for trainingIdx in range(totalTraining):
    del net
    print(str(trainingIdx)+'th training')
    print('==> Building Model...')
    #net=CRNN(outputSize=classNum,hiddenSize=1024)
    net=smcShared(classNum)
    net=net.to(device)
    if device=='cuda' or 'cuda: 0' or 'cuda: 1':
        cudnn.benchmark=True
        #net=nn.DataParallel(net)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9,weight_decay=5e-4)
    for epoch in range(startEpoch,startEpoch+trainingEpoch):
        train(epoch)
        testing(epoch)
    print('current best acc is: ', bestAcc)
