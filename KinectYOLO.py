from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
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

parser = argparse.ArgumentParser()
parser.add_argument("--model_def", type=str, default="config/yolov3plate.cfg", help="path to model definition file")
parser.add_argument("--config_path", type=str, default="config/yolov3plate.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_590.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="TrainingData/plate.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
# 아래의 숫자가 줄수록 class끼리 겹치지않음
parser.add_argument("--nms_thres", type=float, default=1, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("online", exist_ok=True)

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

class RGBShow(object):
    def __init__(self):
        pygame.init()
        self.clock=pygame.time.Clock()
        self.infoObject=pygame.display.Info()
        self.done=False
        self.clock=pygame.time.Clock()
        self.kinect=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        #self.frameSurface=pygame.Surface((496,279),0,32)
    def run(self):
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
                #imgTestPIL=tmpFrame1[:,:,:3]
                cv2.imwrite('./online/saveAndLoad.jpg',tmpFrame1)
                imgTestPIL=cv2.imread('./online/saveAndLoad.jpg')
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
                        print(detections)
                        for x1,y1,x2,y2,conf,cls_conf,cls_pred in detections:
                            try:
                                x1,x2=x1.data.cpu().numpy(),x2.data.cpu().numpy()
                                y1,y2=y1.data.cpu().numpy(),y2.data.cpu().numpy()
                                x1,x2=int(math.floor(x1)),int(math.floor(x2))
                                y1,y2=int(math.floor(y1)),int(math.floor(y2))
                                imgTestPIL=cv2.rectangle(imgTestPIL,(x1,y1),(x2,y2),(255,255,255),3)
                            except:
                                pass
                cv2.imwrite('./online/Trial.jpg',imgTestPIL)
                cv2.imshow('Kinect',imgTestPIL)
            self.clock.tick(60)
        self.kinect.close()
        pygame.quit()

game=RGBShow()
game.run()