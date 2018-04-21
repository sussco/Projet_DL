class Pooling():



    def __init__(self):
        self.chosenOne = [];





    def propagation(self,image, poolSize, poolType):
        x = image.shape[0] % poolSize
        y = image.shape[1] % poolSize
        imageCp = image
        for i in range(x):
            imageCp = np.insert(image, image.shape[1], 0, axis = 1)
        for i in range(y):
            imageCp = np.insert(imageCp,imageCp.shape[0],0, axis = 0)
        output = np.zeros((imageCp.shape[0]/poolSize, imageCp.shape[1]/poolSize))
        self.chosenOne = np.zeros((output.shape[0], output.shape[1], 2))
        print(imageCp)
        i = j = 2
        if poolType == "maximum":
            for i in range(0,image.shape[0], poolSize):
                for j in range(0, image.shape[1], poolSize):
                    output[i/poolSize,j/poolSize] = np.max(imageCp[i: i + poolSize, j : j + poolSize])
                    a = np.argmax((imageCp[i: i + poolSize, j : j + poolSize]))
                    self.chosenOne[i/poolSize,j/poolSize] = [i + a /poolSize ,j + a% poolSize]
            return output
        if poolType == "average":
            for i in range(len(image.shape[0]) - x, poolSize):
                for j in range(len(image.shape[1]) - y, poolSize):
                    output[i,j] = image[i: i + poolSize, j : j + poolSize].sum() / (poolSize**2)
            return output
        if poolType == "quadratic":
            for i in range(len(image.shape[0]) - x, poolSize):
                for j in range(len(image.shape6[1]) - y, poolSize):
                    output[i,j] = math.sqrt(((image[i: i + poolSize, j : j + poolSize])**2).sum())
            return output
        else:
            raise NameError("This pooling function hasn't been implemented yet")

    def backPropagation(self, deltasTable):
        return self.chosenOne

test = np.array([[-1,0,1],[-2,0,4],[-1,0,1]])
print(test)
print(test.shape)
a = Pooling()
print(a.pooling(test,2,"maximum"))
print(a.chosenOne)
