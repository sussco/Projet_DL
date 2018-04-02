import numpy as np

class Layer:

    """ This lass defines a specific layer, we can change the
    number of entries,
    """

    def __init__(self, nbEntries, nbNeurones):
        self.weights = np.random.normal(0, 0.01, [nbNeurones,nbEntries])
        self.biais = np.random.normal(0,0.01,nbNeurones)
        self.a = np.zeros(nbNeurones)



    def propagation(self):
        print("Method undefined in subclass of Layer")

    def backPropagation(self):
        print("Method undefined in subclass of Layer")

    def backPropagation(self):
        print("Method undefined in subclass of Layer")

    def reLU(self):
        print("ReLU fct undefined")
        pass

    def sigmoid(self):
        print("sigmoid Undefined")
