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

    def calVectorForSvm(self,imgList,n_cluster):
        g=Graph(r"E:\ds2018")
        featMat=np.float32([]).reshape((0,128))
        featList=[]
        featVec=np.float32([]).reshape((0,n_cluster))
        for imgPath,type in imgList:
            mat=g.getGreyGraph(imgPath)
            if mat is None:
                continue
            feat=self.getFeature(mat)
            featList.append(feat)
            featMat=np.append(featMat,feat,axis=0)

        kmeans=KMeans(n_cluster)

        kmeans.fit(featMat)

        centers=kmeans.cluster_centers_

        for feature in featList:
            featVec=np.append(featVec,self._calcFeatVec(feature,kmeans,n_cluster,centers),axis=0)


        return (featVec,centers)

    def _calcFeatVec(self,features,kmeans,n_cluster,centers):
        count=kmeans.predict(features)
        re=[]
        for i in range(0,n_cluster,1):
            re.append(count[count==i])

        return np.array(re).reshape([1,-1])