import cv2
import numpy as np
from sklearn import svm
from Graph.Graph import Graph

class SIFT:
    def __init__(self,threshold=200):
        self.threshold=threshold

    def getFeature(self,imgMat):
        sift=cv2.xfeatures2d.SIFT_create(self.threshold)
        kp,des=sift.detectAndCompute(imgMat,None)

        return des
