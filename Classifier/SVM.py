from Graph.Graph import Graph
from Features.SIFT import SIFT
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from libsvm.python.svmutil import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import *


def kMean(clusterNum):
    g=Graph(r"E:\ds2018")
    vocaDir=r"temp/vocabulary/"
    if not os.path.exists(vocaDir):
        os.makedirs(vocaDir)
    sift=SIFT()
    centers=[]
    for i in range(g.getTypeNum()):
        print("[kmeans]:"+str(i))
        imgPaths=g.getTrainSet(i)
        features=np.float32([]).reshape((0,128))
        for imgPath ,type in imgPaths:
            imgMat=g.getGreyGraph(imgPath)
            if imgMat is None:
                print("[kmean]:"+imgPath+" is None")
                continue
            feature=sift.getFeature(imgMat)
            features=np.append(features,feature,axis=0)


        kmeans= KMeans(n_clusters=clusterNum).fit(features)
        filename=os.path.join(vocaDir,str(i)+".npy")
        np.save(filename,kmeans.cluster_centers_)

        centers.append(kmeans.cluster_centers_)

    return centers

def calcFeatVec(features,centers,clusterNum):
    featVec = np.zeros((1, clusterNum))
    for i in range(0, features.shape[0]):
        fi = features[i]
        y=np.arange(clusterNum)
        mat,_=np.meshgrid(fi,y)
        diffMat = mat - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        sortedIndices = sqSum.argsort()
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
trainTypes=np.int32([])
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
        trainTypes=np.append(trainTypes,np.int32([type]))

print(trainTypes.shape)
trainTypes=trainTypes.astype(np.float32)


tuned_parameters=[
    {
        'kernel':['rbf'],
        'gamma':[1e-1,0.05,1e-2,1e-3,1e-4],
        'C':[1,3,5,10,15,100,1000]
    }
]
svm=GridSearchCV(SVC(decision_function_shape='ovo'),tuned_parameters,cv=5)
svm.fit(trainData,trainTypes)
print(svm.best_params_)

svm=svm.best_estimator_

print("done")

#classify
testList=g.getTestSet()
scoreData=np.float32([]).reshape((0,clusterNum))
scoreType=np.float32([]).reshape((0,1))
for imgPath,type in testList:
    sift=SIFT()
    imgMatrix=g.getGreyGraph(imgPath)
    if imgMatrix is None:
        print("[test]:"+imgPath)
        continue
    siftFeature=sift.getFeature(imgMatrix)
    featVec=calcFeatVec(siftFeature,centers[type],clusterNum)
    scoreData=np.append(scoreData,featVec,axis=0)
    scoreType=np.append(scoreType,np.float32([type]).reshape((1,1)),axis=0)


print(svm.score(scoreData,scoreType))