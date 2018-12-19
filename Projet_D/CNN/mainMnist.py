from Net import *
from imageReader import *
import random
import pickle
# from augmentation import generateImageCifar

#################### MNIST DATASET ######################
labelled_images = list_labelled_images2D('MNIST/train-images-idx3-ubyte', 'MNIST/train-labels-idx1-ubyte', 50000, 0, 'digits')
test_images = list_labelled_images2D('MNIST/t10k-images-idx3-ubyte', 'MNIST/t10k-labels-idx1-ubyte', 10000, 0, 'digits')

################### EMNIST DATASET ######################
# labelled_images = list_labelled_images2D('EMNIST/emnist-letters-train-images-idx3-ubyte', 'EMNIST/emnist-letters-train-labels-idx1-ubyte', 120000, 0, 'letters')
# test_images = list_labelled_images2D('EMNIST/emnist-letters-test-images-idx3-ubyte', 'EMNIST/emnist-letters-test-labels-idx1-ubyte', 10000, 0, 'letters')

################### CIFAR10 DATASET #####################
# batch = []
# batch.append(get_data('cifar-10-batches-py/data_batch_1'))
# batch.append(get_data('cifar-10-batches-py/data_batch_2'))
# batch.append(get_data('cifar-10-batches-py/data_batch_3'))
# batch.append(get_data('cifar-10-batches-py/data_batch_4'))
# batch.append(get_data('cifar-10-batches-py/data_batch_5'))
# tests = get_data('cifar-10-batches-py/test_batch')

# fRes1= open("Tests/MNIST/Mnist_CNN_batch100","ab")


nets = []
lr = 0.1
# pick = pickle.load(open('CIFAR10_1batch4', 'rb'))

for k in range(1):
        neuralNet = Net()
        # for l in range(len(batch)):
        neuralNet.train(labelled_images[0], labelled_images[1], 10, lr)
        # print(10000, l)
        print(k)
        nets.append( (neuralNet.test(test_images[0], test_images[1]), neuralNet, lr))
        lr += 0.1

print(nets)
pickle.dump(nets,fRes1)
