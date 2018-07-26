from skimage import feature
import numpy as np
import cv2
from Graph.Graph import Graph

class LBP:

    '''
    ps:list
    rs:list
    methods:list    ("default","ror","uniform","var")
    '''
    def __init__(self,ps,rs,method="default"):
        self.p=ps
        self.r=rs
        self.method=method

    def _calcVec(self,feat):
        re = np.zeros(self.getVecLength())
        unique, counts = np.unique(feat, return_counts=True)
        unique=unique.astype(np.int32)
        for i in range(unique.size):
            re[unique[i]] = counts[i]

        return re

    def getVecLength(self):
        return 2**8

    def getFeature(self,imgMat):

        feat=feature.local_binary_pattern(imgMat,8,1,method='uniform')
        return self._calcVec(feat).reshape((1,-1))

    def getFeatVecForSvm(self,imgList,load=0):
        if load==1:
            feats=np.load(r"temp/featVectLbp.npy")
            return feats

        g=Graph(r"E:\ds2018")
        feats=np.float32([]).reshape((0,self.getVecLength()))
        i=0
        for imgPath,type in imgList:
            print("[lbp]:"+str(i))
            i+=1

            mat=g.getGreyGraph(imgPath)
            if mat is None:
                continue
            feat=self.getFeature(mat)
            feats=np.append(feats,feat.reshape((1,-1)),axis=0)

        np.save(r"temp/featVectLbp.npy", feats)
        return feats

'''
g=Graph(r"E:\ds2018")
trainList=g.getTrainSet()
lbp=LBP([8],[1])
feats=lbp.getFeatVecForSvm(trainList)
print(feats.shape)
'''

