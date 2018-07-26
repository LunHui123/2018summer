import os
import cv2
import random
import numpy as np
import shutil

'''
目录结构组织为(divide前):
.../
    rootDir/
            originDir/
                    bear/
                    bicycle/
                    bird/
                    ...

目录结构组织为(divide之后)：
.../
    rootDir/
            originDir/
                    bear/
                    bicycle/
                    bird/
                    ...
            train/
                    bear/
                    bicycle/
                    bird/
                    ...
            test/
            originCSV.csv
            trainCSV.csv
            testCSV.csv
'''
class Graph:

    def __init__(self,rootDir):
        self.rootDir=rootDir
        self.originCsvName="originCSV.csv"
        self.trainCsvName="trainCSV.csv"
        self.testCsvName="testCSV.csv"
        self.type=('bear','bicycle','bird','car','cow','elk','fox',
                   'giraffe','horse','koala','lion','monkey','plane',
                   'puppy','sheep','statue','tiger','tower','train',
                   'whale','zebra')

    def _writeToCSV(self,l,path):

        file=open(path,"w+")
        for i in l:
            file.write(str(i[0])+','+str(i[1])+"\n")

        file.close()

    def _readCsv(self,path):
        csv = open(path, "r")
        lines = csv.readlines()
        reList = []
        for i in lines:
            i = i.strip('\n')
            (path, type) = i.split(',')
            reList.append([path, int(type)])

        csv.close()
        return reList

    def _copyAndGetList(self,l,dir):
        trainImgList = []
        for i in l:
            fileName = os.path.split(i[0])[1]
            newPath = os.path.join(dir, fileName)
            trainImgList.append([newPath, i[1]])

            shutil.copy(i[0],newPath)

        return trainImgList

    def _normGraph(self,path,h,w):

        mat=cv2.imread(path)
        #print("path: "+path)

        if  mat is None:
            return

        #print(mat.shape)

        mat=cv2.resize(mat, (h,w),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path,mat)

    def _toCsvList(self,l,type,dir):
        ll=[]
        for i in l:
            ll.append([os.path.join(dir,i),type])
        return ll

    def getTypeId(self,typeStr):
        if self.type.count(typeStr) <=0:
            return -1
        return self.type.index(typeStr)

    def getTypeName(self,id):
        if id<0 or id >=len(self.type):
            return None

        return self.type[id]

    def getTypeNum(self):
        return len(self.type)

    def normTrainTest(self,h,w):
        trainList=self.getTrainSet()
        testList=self.getTestSet()
        for i in trainList:
            self._normGraph(i[0],h,w)
        for i in testList:
            self._normGraph(i[0],h,w)

    def isDivided(self):
        if os.path.exists(os.path.join(self.rootDir,self.trainCsvName)):
            return True

    def getGraph(self,path):
        [dirname,filename]=os.path.split(path)

        matrix=cv2.imread(path)

        return matrix

    def getGreyGraph(self,path):
        matrix = cv2.imread(path)
        if matrix is None:
            print("read error")
            return None

        matrix=cv2.cvtColor(matrix,cv2.COLOR_BGR2GRAY)

        return matrix

    def divideTrainTest(self,originName,trainSize=4/5):
        originDirPath=os.path.join(self.rootDir,originName)
        if(not (os.path.isdir(originDirPath) and os.path.exists(originDirPath)) ):
            print("error dirPath")
            return

        rootDir=self.rootDir
        originCSV=os.path.join(rootDir,self.originCsvName)
        trainCSV=os.path.join(rootDir,self.trainCsvName)
        testCSV=os.path.join(rootDir,self.testCsvName)

        trainDirPath=os.path.join(rootDir,r"train")
        testDirPath=os.path.join(rootDir,r"test")


        if os.path.exists(trainDirPath):
            shutil.rmtree(trainDirPath)
        if os.path.exists(testDirPath):
            shutil.rmtree(testDirPath)
#        os.mkdir(trainDirPath)
        os.mkdir(testDirPath)


        #print(fileDirs)

        shutil.copytree(originDirPath,trainDirPath)

        typeDirs=os.listdir(trainDirPath)
        testList=[]
        trainList=[]
        for type in typeDirs:
            typeDir=os.path.join(trainDirPath,type)

            imgs=os.listdir(typeDir)
            trainImgs=random.sample(imgs,int(len(imgs)*trainSize))
            testImgs=[i for i in imgs if i not in trainImgs]

            for imgName in testImgs:
                oldImgPath=os.path.join(typeDir,imgName)
                newImgPath=os.path.join(testDirPath,imgName)
                shutil.move(oldImgPath,newImgPath)

            testList.extend(self._toCsvList(testImgs,self.getTypeId(type),testDirPath))
            trainList.extend(self._toCsvList(trainImgs,self.getTypeId(type),typeDir))

        self._writeToCSV(testList,testCSV)
        self._writeToCSV(trainList,trainCSV)

        '''
        typeDirs=os.listdir(originDirPath)
        imgPathList=[]

        for type in typeDirs:
            typeDir=os.path.join(originDirPath,type)
            imgNames=os.listdir(typeDir)

            for imgName in imgNames:
                imgPathList.append([os.path.join(typeDir,imgName),type])

        self._writeToCSV(imgPathList,originCSV)
        
        random.shuffle(imgPathList)
        trainList=random.sample(imgPathList,int(len(imgPathList)*trainSize))
        testList=[i for i in imgPathList if i not in trainList]

        trainImgList=self._copyAndGetList(trainList,trainDirPath)
        testImgList=self._copyAndGetList(testList,testDirPath)

        self._writeToCSV(trainImgList,trainCSV)
        self._writeToCSV(testImgList,testCSV)
        '''


        return [trainList,testList]

    #返回训练集图像的路径与类型
    #list [[path,type] ...]
    def getTrainSet(self,type=-1):

        trainCsvPath=os.path.join(self.rootDir,self.trainCsvName)
        res=self._readCsv(trainCsvPath)

        if type==-1:
            return res
        else:
            res=[i for i in res if i[1]==type]

        return res


    # 返回测试集图像的路径与类型
    # list [[path,type] ...]
    def getTestSet(self):
        testCsvPath = os.path.join(self.rootDir, self.testCsvName)

        return self._readCsv(testCsvPath)


'''
g=Graph(r"E:\ds2018")
#g.divideTrainTest("ds2018")
g.normTrainTest(64,64)


count=0
for path,type in g.getTrainSet():
    if not os.path.exists(path):
        print(path+" not exists")

    if cv2.imread(path) is None:
        print(path+" is None")

    count+=1
print(count)
'''