import numpy as np
import cv2
from scipy.spatial import distance
def warpPerspectiveMine(x,y,transformationMatrix):
    a=transformationMatrix[0,0]*x+transformationMatrix[0,1]*y+transformationMatrix[0,2]
    b=transformationMatrix[2,0]*x+transformationMatrix[2,1]*y+transformationMatrix[2,2]
    c=transformationMatrix[1,0]*x+transformationMatrix[1,1]*y+transformationMatrix[1,2]
    return (int(a/b),int(c/b))

def transformAfterWarp(img,detectedObject,H):
    ## extracting the vertex of the white place
    for objIdx in range(len(detectedObject)):
        originCoord=detectedObject[objIdx][0]
        label=detectedObject[objIdx][1]
        convertedCoord=warpPerspectiveMine(originCoord[0],originCoord[1],H)
        detectedObject[objIdx][0]=convertedCoord
    return detectedObject

def distanceMine(a,b):
    return int(distance.euclidean(a,b))
def selectingPos(height,weight,detectedObject):
    distIdx=[]
    leftPosX=(0+weight/2)/2
    leftPosY=(height/2+height)/2
    leftPos=[leftPosX,leftPosY]
    rightPosX=(weight+weight/2)/2
    rightPosY=(height/2+height)/2
    rightPos=[rightPosX,rightPosY]
    forwardPosX=(0+height/2)/2
    forwardPosY=(height/2+height)/2
    forwardPos=[forwardPosX,forwardPosY]
    dist=np.zeros((len(detectedObject),3))
    for objIdx in range(len(detectedObject)):
        dist[objIdx,0]=distanceMine(leftPos,detectedObject[objIdx][0])
        dist[objIdx,1]=distanceMine(rightPos,detectedObject[objIdx][0])
        dist[objIdx,2]=distanceMine(forwardPos,detectedObject[objIdx][0])
    # nearest object idx from the predefined 'Left'
    tmp=np.where(dist == min(dist[:, 0]))[0]
    distIdx.append(tmp[0])
    # nearest object idx from the predefined 'Right'
    tmp=np.where(dist==min(dist[:,1]))[0]
    distIdx.append(tmp[0])
    # nearest object idx from the predefined 'Forward'
    tmp=np.where(dist==min(dist[:,2]))[0]
    distIdx.append(tmp[0])
    leftObj=detectedObject[distIdx[0]]
    rightObj=detectedObject[distIdx[1]]
    forwardObj=detectedObject[distIdx[2]]
    obj=[leftObj,rightObj,forwardObj]
    return obj