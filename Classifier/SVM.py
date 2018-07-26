from Graph.Graph import Graph
from Features.SIFT import SIFT
import numpy as np
import cv2
import os
from sklearn.svm import SVC,SVR
from libsvm.python.svmutil import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import *
from Features.LBP import LBP
from Features.HOG import HOG


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
clusterNum=21
'''
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

'''
print("calc kmeans")
lbp=LBP([8],[1])
hog=HOG()
imgList=g.getTrainSet()
trainTypes,trainDataSift,centers=sift.calcVectorForSvm(imgList,clusterNum,1)
print("calc kmeans done")

print("calc lbp")
trainDataLbp=lbp.getFeatVecForSvm(imgList,1)
print("calc lbp done")

print("calc hog")
trainDataHog=hog.getFeatVecForSvm(imgList,1)
print("calc hog done")

trainData=np.append(trainDataSift,trainDataLbp,axis=1)
trainData=np.append(trainData,trainDataHog,axis=1)
#trainData=trainDataLbp
print("train type shape:"+str(trainTypes.shape))
print("train data shape"+str(trainData.shape))

tuned_parameters=[
    {
        'kernel':['rbf'],
        'gamma':[1e-3,5e-3,10e-3],
        'C':[1,3,5,10,15,100,1000]
    }
]
'''
tuned_parameters=[
    {
        'kernel':['linear'],
        'C':[1,3,5,10,15,100,1000]
    }
]
'''
#svm=GridSearchCV(SVC(decision_function_shape='ovo'),tuned_parameters,cv=5)
svm=SVC(C=5,gamma=0.0001,kernel='linear')
svm.fit(trainData,trainTypes)
#print(svm.best_params_)

#svm=svm.best_estimator_

print("train done")

#classify
testList=g.getTestSet()
scoreData=np.float32([]).reshape((0,clusterNum+lbp.getVecLength()+hog.getVecLength()))
scoreType=np.float32([]).reshape((0,1))
for imgPath,type in testList:
    #print("[test]:"+imgPath +"  "+g.getTypeName(type))
    sift=SIFT()
    imgMatrix=g.getGreyGraph(imgPath)
    if imgMatrix is None:
        print("[test]:"+imgPath)
        continue
    siftFeature=sift.getFeature(imgMatrix)
    featVecSift=sift.calcFeatVec(siftFeature,centers,clusterNum)
    featVecLbp=lbp.getFeature(imgMatrix)
    featVecHog=hog.getFeature(imgMatrix)
    featVec=np.append(featVecSift.reshape((1,-1)),featVecLbp.reshape((1,-1)),axis=1)
    #featVec=featVecLbp
    featVec=np.append(featVec,featVecHog,axis=1)

    scoreData=np.append(scoreData,featVec,axis=0)
    scoreType=np.append(scoreType,np.float32([type]).reshape((1,1)),axis=0)


print(svm.score(scoreData,scoreType))