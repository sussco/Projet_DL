import numpy as np

class Pool():



    def __init__(self, poolSize):
        self.poolSize = poolSize





    def propagation(self,image):
        poolSize = self.poolSize
        x = image.shape[1] % poolSize
        y = image.shape[2] % poolSize
        imageCp = image
        for i in range(x):
            imageCp = np.insert(image, image.shape[2], 0, axis = 2)
        for i in range(y):
            imageCp = np.insert(imageCp,imageCp.shape[1],0, axis = 1)
        output = np.zeros((imageCp.shape[0] ,int(imageCp.shape[1]/poolSize), int(imageCp.shape[2]/poolSize)))
        self.chosenOne = np.zeros((imageCp.shape[0] ,imageCp.shape[1], imageCp.shape[2]))
        for depth in range(image.shape[0]):
            for i in range(0,image.shape[1], poolSize):
                for j in range(0, image.shape[2], poolSize):
                    output[depth, int(i/poolSize),int(j/poolSize)] = np.max(imageCp[depth, i: i + poolSize, j : j + poolSize])
                    a = np.unravel_index(imageCp[depth, i: i + poolSize, j : j + poolSize].argmax(), imageCp[depth, i: i + poolSize, j : j + poolSize].shape)
                    self.chosenOne[depth, i+a[0]%poolSize, j + a[1]% poolSize] = 1
        return output

    def backPropagation(self, deltaTable):
        poolSize = self.poolSize
        for depth in range(deltaTable.shape[0]):
            for i in range(deltaTable.shape[1]):
                for j in range(deltaTable.shape[2]):
                    self.chosenOne[depth, i*poolSize: i*poolSize+poolSize, j*poolSize: j*poolSize+poolSize] = self.chosenOne[depth, i*poolSize: i*poolSize+poolSize, j*poolSize: j*poolSize+poolSize]*deltaTable[depth,i,j]
        return self.chosenOne
