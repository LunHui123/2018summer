import cv2
import numpy as np
from sklearn import svm
from Graph.Graph import Graph
from sklearn.cluster import *

class SIFT:
    def __init__(self,threshold=200):
        self.threshold=threshold

    def getFeature(self,imgMat):
        sift=cv2.xfeatures2d.SIFT_create(self.threshold)
        kp,des=sift.detectAndCompute(imgMat,None)

        return des

    def calcVectorForSvm(self,imgList,n_cluster,load=0):
        if load==1:
            types=np.load(r"temp/types.npy")
            featVec=np.load(r"temp/featVectSift.npy")
            centers=np.load(r"temp/centersSift.npy")
            return  (types,featVec,centers)

        g=Graph(r"E:\ds2018")
        featMat=np.float32([]).reshape((0,128))
        featList=[]
        featVec=np.float32([]).reshape((0,n_cluster))
        types=np.float32([])
        i=0
        for imgPath,type in imgList:
            print("[kmeans before]:"+str(i))
            i+=1
            mat=g.getGreyGraph(imgPath)
            if mat is None:
                continue
            feat=self.getFeature(mat)
            featList.append(feat)
            featMat=np.append(featMat,feat,axis=0)
            types=np.append(types,np.float32([type]))

        kmeans=KMeans(n_cluster)

        kmeans.fit(featMat)

        centers=kmeans.cluster_centers_

        i=0
        for feature in featList:
            print("[kmeans after]:" + str(i))
            i+=1
            featVec=np.append(featVec,self._calcFeatVec(feature,kmeans,n_cluster,centers),axis=0)

        np.save(r"temp/types.npy",types)
        np.save(r"temp/featVectSift.npy",featVec)
        np.save(r"temp/centersSift.npy",centers)

        return (types,featVec,centers)

    def _calcFeatVec(self,features,kmeans,n_cluster,centers):
        count=kmeans.predict(features)

        re=np.zeros(n_cluster)
        unique,counts=np.unique(count,return_counts=True)
        for i in range(unique.size):
            re[unique[i]]=counts[i]

        return re.reshape((1,-1))

    def calcFeatVec(self,features,centers,n_cluster):
        featVec = np.zeros((1, n_cluster))
        for i in range(0, features.shape[0]):
            fi = features[i]
            y = np.arange(n_cluster)
            mat, _ = np.meshgrid(fi, y)
            diffMat = mat - centers
            sqSum = (diffMat ** 2).sum(axis=1)
            sortedIndices = sqSum.argsort()
            idx = sortedIndices[0]  # index of the nearest center
            featVec[0][idx] += 1
        return featVec