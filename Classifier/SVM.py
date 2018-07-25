from Graph.Graph import Graph
from Features.SIFT import SIFT
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from libsvm.python.svmutil import *


def kMean(clusterNum):
    g=Graph(r"E:\ds2018")
    vocaDir=r"temp/vocabulary/"
    if not os.path.exists(vocaDir):
        os.makedirs(vocaDir)
    sift=SIFT()
    centers=[]
    for i in range(g.getTypeNum()):
        imgPaths=g.getTrainSet(i)
        features=np.float32([]).reshape((0,128))
        for imgPath ,type in imgPaths:
            imgMat=g.getGreyGraph(imgPath)
            if imgMat is None:
                print("[kmean]:"+imgPath+" is None")
                continue
            feature=sift.getFeature(imgMat)
            features=np.append(features,feature,axis=0)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        print(i)
        print(features.shape)
        compactness, label, center = cv2.kmeans(features, clusterNum, None,criteria, 20, flags)

        filename=os.path.join(vocaDir,str(i)+".npy")
        np.save(filename,(label,center))

        centers.append(center)

    return centers

def calcFeatVec(features,centers,clusterNum):
    featVec = np.zeros((1, clusterNum))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (clusterNum, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]  # index of the nearest center
        featVec[0][idx] += 1
    return featVec

def loadVoca():
    g=Graph(r"E:\dataset")
    vocaDir = r"temp/vocabulary/"
    centers=[]
    for i in range(g.getTypeNum()):
        filePath=os.path.join(vocaDir,str(i)+'.npy')
        _,center=np.load(filePath)
        centers.append(centers)

    return centers


print("start")
g=Graph(r"E:\ds2018")
if not g.isDivided():
    g.divideTrainTest("ds2018")

sift=SIFT()
clusterNum=20

print("calc kmeans")
centers=kMean(clusterNum)

#centers=loadVoca()
trainData=np.float32([]).reshape(0,clusterNum)
trainTypes=np.int32([]).reshape(0,1)
flag=False

print("prepare trainData")
for i in range(g.getTypeNum()):
    print(i)
    trainList=g.getTrainSet(i)
    for imgPath ,type in trainList:
        print("[trainData]:"+imgPath)
        mat=g.getGreyGraph(imgPath)
        if mat is None:
            print(imgPath+"is None")
            continue
        feature=sift.getFeature(mat)
        featVec=calcFeatVec(feature,centers[type],clusterNum)
        trainData=np.append(trainData,featVec,axis=0)
        trainTypes=np.append(trainTypes,np.int32([type]).reshape((1,1)),axis=0)

print(trainTypes.shape)
#svm=cv2.ml.SVM_create()
svm=SVC()
trainTypes.reshape((-1,1))
trainTypes=trainTypes.astype(np.float32)
svm.fit(trainData,trainTypes)
#svm.save("svm.clf")

print("done")

#classify
total=0
correct=0

testList=g.getTestSet()
for imgPath,type in testList:
    sift=SIFT()
    imgMatrix=g.getGreyGraph(imgPath)
    if imgMatrix is None:
        print("[test]:"+imgPath)
        continue
    siftFeature=sift.getFeature(imgMatrix)
    featVec=calcFeatVec(siftFeature,centers[type],clusterNum)
    total+=1
    pred=svm.predict(featVec)
    if type==int(pred):
        correct+=1

accu=correct/total

print(accu*100)