from Net import *
import imageReader



labelled_images = imageReader.list_labelled_images2D('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 200, 0, 'digits')
test_images = imageReader.list_labelled_images2D('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 100, 0, 'digits')


neuralNet = Net()

neuralNet.train(labelled_images[0], labelled_images[1], 1, 0.1)
neuralNet.test(test_images[0], test_images[1])
