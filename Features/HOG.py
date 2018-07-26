from skimage import feature as ft
from Graph.Graph import Graph
import numpy as np

class HOG:

    def getVecLength(self):
        return 324

    def getFeature(self,imgMat):
        feat=ft.hog(imgMat,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2))
        return feat.reshape((1,-1))

    def getFeatVecForSvm(self,imgList,load=0):

        if load==1:
            feats=np.load(r"temp/featVectHog.npy")
            return feats

        g=Graph(r"E:\ds2018")

        feats=np.float32([]).reshape((0,self.getVecLength()))
        for imgPath,type in imgList:
            mat=g.getGreyGraph(imgPath)
            if mat is None:
                continue
            feat=self.getFeature(mat)
            feats=np.append(feats,feat,axis=0)


        np.save(r"temp/featVectHog.npy", feats)
        return feats

'''
g=Graph(r"E:\ds2018")
hog=HOG()
print(hog.getFeatVecForSvm(g.getTrainSet()))
'''