import numpy as np

class Layer:
    
    """ This lass defines a specific layer, we can change the 
    number of entries,
    """
    
    def __init__(self, nbEntries, nbNeurones):
        self.weights = np.random.normal(0, 0.01, [nbNeurones,nbEntries])
        self.biais = np.random.normal(0,0.01,nbNeurones)
        self.a = np.zeros(nbNeurones)
    
    def print(self):
        print(self.weights)