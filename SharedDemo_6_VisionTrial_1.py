from Model.YoloModels import *
from utils.utils import *
from utils.datasets import *
from utils.CoordinateTransform import *

import os
import sys
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import ctypes
import _ctypes
import pygame

import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--model_def", type=str, default="Vision/config/yolov3plate.cfg", help="path to model definition file")
parser.add_argument("--config_path", type=str, default="Vision/config/yolov3plate.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="Vision/checkpoints/yolov3_ckpt_590.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="Vision/TrainingData/plate.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
# ?븘?옒?쓽 ?닽?옄媛? 以꾩닔濡? class?겮由? 寃뱀튂吏??븡?쓬
parser.add_argument("--nms_thres", type=float, default=1, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("Vision/online", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

horizonList=[]
count=0

tmpArray=[]
detectedObject=[]
def mouseCallback(event,x,y,flags,param):
    global horizonList,count,verticalList
    if event==cv2.EVENT_LBUTTONDOWN and count<4:
        a=(x,y)
        a=list(a)
        horizonList.append(a)
        count+=1
class RGBShow(object):
    def __init__(self,Hheights,Hweights):
        pygame.init()
        self.infoObject=pygame.display.Info()
        self.done=False
        self.kinect=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.Hheights=Hheights
        self.Hweights=Hweights
        #self.frameSurface=pygame.Surface((496,279),0,32)
    def workspace(self):
        global horizonList,count,tmpArray
        self.done=False
        while not self.done:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    self.done=True
                elif event.type==pygame.VIDEORESIZE:
                    self.screen=pygame.display.set_mode(event.dict['size'],
                                                        pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE,32,0)
            if self.kinect.has_new_color_frame():
                frame=self.kinect.get_last_color_frame()
                tmpFrame1=frame.reshape((1080,1920,4))
                cv2.imshow('set workspace',tmpFrame1)
                if count<4:
                    cv2.setMouseCallback('set workspace',mouseCallback)
                else:
                    tmpArray=np.array(horizonList)
                    horizonPts = np.float32(
                        [horizonList[0], horizonList[1], horizonList[2], horizonList[3]])
                    ptsHorizon = np.float32([[0,0],[self.Hweights,0],[0,self.Hheights],[self.Hweights,self.Hheights]])
                    H=cv2.getPerspectiveTransform(horizonPts,ptsHorizon)
                    cv2.destroyAllWindows()
                    return H


    def run(self,H,xRatio,yRatio,userIntention):
        global detectedObject,heights, weights, horizonList,tmpArray
        self.done=False
        while not self.done:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    self.done=True
                elif event.type==pygame.VIDEORESIZE:
                    self.screen=pygame.display.set_mode(event.dict['size'],
                                                        pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE,32,0)
            if self.kinect.has_new_color_frame():
                detectedObject=[]
                frame=self.kinect.get_last_color_frame()
                tmpFrame1=frame.reshape((1080,1920,4))
                #imgTestPIL=tmpFrame1[:,:,:3]
                cv2.imwrite('./Vision/online/saveAndLoad.jpg',tmpFrame1)
                imgTestPIL=cv2.imread('./Vision/online/saveAndLoad.jpg')
                imgTest=transforms.ToTensor()(imgTestPIL)
                imgTest,_=pad_to_square(imgTest,0)
                imgTest=resize(imgTest,416)
                imgTest=Variable(imgTest.type(Tensor))
                with torch.no_grad():
                    imgTest=imgTest[np.newaxis,:,:,:]
                    detections=model(imgTest)
                    detections=non_max_suppression(detections,conf_thres=0.8)
                img_detections.extend(detections)
                for imgIdx, (detections) in enumerate(img_detections):
                    if detections is not None:
                        detections = rescale_boxes(detections,416,imgTestPIL.shape[:2])
                        try:
                            for x1,y1,x2,y2,conf,cls_conf,cls_pred in detections:
                                ## only perform object detection located on the workspace
                                if min(tmpArray[:,1])<(y1+y2)/2<max(tmpArray[:,1]) and min(tmpArray[:,0])<(x1+x2)/2<max(tmpArray[:,0]):
                                    x1,x2=x1.data.cpu().numpy(),x2.data.cpu().numpy()
                                    y1,y2=y1.data.cpu().numpy(),y2.data.cpu().numpy()
                                    x1,x2=int(math.floor(x1)),int(math.floor(x2))
                                    y1,y2=int(math.floor(y1)),int(math.floor(y2))
                                    imgTestPIL=cv2.rectangle(imgTestPIL,(x1,y1),(x2,y2),(255,255,255),3)
                                    #imgTestPIL[y1:y2,x1:x2,:]=0
                                    axis=[math.floor(x1+(x2-x1)*xRatio),math.floor(y1+(y2-y1)*yRatio)]
                                    imgTestPIL[axis[1],axis[0],:]=0
                                    if cls_pred==0:
                                        axis=[axis,'ball']
                                        detectedObject.append(axis)
                                    elif cls_pred==1:
                                        axis=[axis,'cup']
                                        detectedObject.append(axis)
                                    elif cls_pred==2:
                                        axis=[axis,'bottle']
                                        detectedObject.append(axis)
                        except:
                            pass
                HResult=cv2.warpPerspective(imgTestPIL,H,(self.Hweights,self.Hheights))
                #HResult=cv2.circle(HResult,(0,0),5,255,-1)
                if len(detectedObject)==0 or userIntention>=3:
                    HResult=cv2.circle(HResult,(int(weights/2),int(heights/2)),5,255,-1)
                    cv2.imshow('warped Horizontal',HResult)
                    cv2.imshow('Kinect',imgTestPIL)
                    self.done=True
                else:
                    detectedObject=transformAfterWarp(HResult,detectedObject,H)
                    selectedObject=selectingPos(heights,weights,detectedObject)
                    print(selectedObject[userIntention])
                    HResult=cv2.circle(HResult,selectedObject[userIntention][0],5,255,-1)
                    cv2.imshow('warped Horizontal',HResult)
                    cv2.imshow('Kinect',imgTestPIL)
                    self.done=True
        self.kinect.close()


heights=91*5
weights=106*5
H=[]
game=RGBShow(heights,weights)
def getPerspectiveMine():
    global H,game
    H=game.workspace()
def mainRunning(userIntention):
    global H
    game1=RGBShow(heights,weights)
    #0:left 1:right 2:forward
    game1.run(H,0.5,0.9,int(userIntention))

