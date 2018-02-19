from Layer import Layer
import numpy as np
import math

class Perceptron:
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def vector_sigmoid(self, x):
        for i in range(len(x)):
            x[i] = self.sigmoid(x[i])
        return x
    
    
    """ Entrées ici"""
    
    """ Couche de sortie """
    
    
    """ Couche cachée / intermédiaires """

    """ Initialisation d'un perceptron"""
    """ La premiere couche correspont à la premiere couche caché qui prend l'entrée
        en paramètre
        les autres sauf la dernière sont des couches cachées"""
        
    def __init__(self, nbHiddenLayers, nbNeuPerLayer , nbOutPuts, nbInputs, learningRate, importance):
        self.learningRate = learningRate
        self.importance = importance
        """ couches cachées """
        self.layers = []
        """ Premiere couche """ 
        self.layers.append(Layer.Layer(nbInputs, nbNeuPerLayer))
        for i in range(0, nbHiddenLayers - 1):
            self.layers.append(Layer.Layer(nbNeuPerLayer, nbNeuPerLayer))
        """ Dernière couche (visible)"""
        self.layers.append(Layer.Layer(nbNeuPerLayer, nbOutPuts))

            
    
    """ Propagation """
    
    def propagation(self, layIn):
        self.layers[0].a = self.vector_sigmoid(np.matmul(self.layers[0].weights, layIn) 
                    + self.layers[0].biais)
        for i in range(1,len(self.layers)):
            self.layers[i].a = self.vector_sigmoid(np.matmul(self.layers[i].weights,self.layers[i-1].a) + self.layers[i].biais)
        return self.layers[-1].a
       
    def fPrime(self, layer):
        f = np.zeros(len(self.layers[layer].a))
        for i in range(0,len(self.layers[layer].a)):
            f[i] = (self.layers[layer].a[i]*(1-self.layers[layer].a[i]))
        return f
                
    def outPutError(self, expectedOutPut):
        return (-(expectedOutPut - self.layers[-1].a) * self.fPrime(len(self.layers)- 1))
    
    def partialDerivate(self, layer, outPutErr):
        if (layer == len(self.layers) - 1):
            return outPutErr
        else:
            return np.matmul(np.transpose(self.layers[layer+ 1].weights), self.partialDerivate(layer + 1,outPutErr
                 )) * self.fPrime(layer)
   
    def layerError(self, layer, outPutErr, layIn):
        delta = self.partialDerivate(layer, outPutErr)
        if(layer == 0): actCouchePrec = np.transpose(layIn)
        else: actCouchePrec = np.transpose(self.layers[layer - 1].a)
        return [np.outer(delta, actCouchePrec), delta]
         
    """ Retropropagation """
        
    
    def backpropagation(self, layIn, expectedOutPut, nbTrainings, weightsTable, biaisTable):
        """ On va utiliser costs et mettre à jour les poids dedans, puis à partir
        de cost on va mettre à jour les poids dansperception
        """
    
        error = self.outPutError(expectedOutPut)
        for l in range(0,len(self.layers)):
            """ On calcule les deux erreur de biais et de poids en même temps"""
            coupleError = self.layerError(l, error, layIn)
            weightsTable[l] += coupleError[0]
            biaisTable[l] += coupleError[1]
            
                
    
    def updateParams(self, weightsTable, biaisTable, nbTrainings):
        """update parameters"""
        for l in range (0,len(self.layers)):
            self.layers[l].weights -= self.learningRate * ((1/nbTrainings)*weightsTable[l]
                                        + self.importance)
            self.layers[l].biais -= self.learningRate * ((1/nbTrainings) * biaisTable[l])
            print("test")
                 




test = Perceptron(0,3,10,10, 100,0)
exp = [0 for i in range(10)]
exp[0] = 1
print(exp)
weightsTable = []
biaisTable = []
for lay in test.layers:
    weightsTable.append(np.zeros([len(lay.weights), len(lay.weights[0])]))
    biaisTable.append(np.zeros([len(lay.weights)]))


exp[0] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[1] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[2] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[3] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[4] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[5] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[6] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[7] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[8] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
exp = [0 for i in range(10)]
exp[9] = 1
test.propagation(exp)
test.backpropagation(exp, exp,10,weightsTable, biaisTable)
test.updateParams(weightsTable, biaisTable, 10)

#for lay in test.layers:
#    print("ligne : ", len(lay.weights), "colonne : ",len(lay.weights[0]),
#          "sorties : ",len(lay.a))
#print(np.matmul(np.transpose(test.layers[2].weights),test.partialDerivate(2,exp)))
test.propagation([1,0,0,0,0,0,0,0,0,0])
#print(test.layers[0].weights)
print(test.layers[-1].a)
print()
#print(weightsTable)
print(test.layerError(1,test.outPutError(exp), exp)[0])
#print(test.lOut)
#print test.loss_last_layer(exp)