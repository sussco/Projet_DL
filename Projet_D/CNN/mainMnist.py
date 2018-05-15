from Net import *
import imageReader
import random
import pickle

# labelled_images = imageReader.list_labelled_images2D('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 60000, 0, 'digits')
# test_images = imageReader.list_labelled_images2D('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 10000, 0, 'digits')

fRes1= open("CIFARLeNet2","ab")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(file):
    labelled_images = unpickle(file)
    print('getting data from file', file, '... ', end = '')
    X = np.asarray(labelled_images[b'data']).astype("uint8")
    X = np.reshape(X, (10000,3,32,32))
    # X = X.transpose([0, 2, 3, 1])
    X = X/float(255)
    Yraw = np.asarray(labelled_images[b'labels'])
    Y = np.zeros((10000,10))
    for i in range(10000):
        Y[i,Yraw[i]] = 1
    print('done')
    return X,Y


batch = []
batch.append(get_data('cifar-10-batches-py/data_batch_1'))
batch.append(get_data('cifar-10-batches-py/data_batch_2'))
batch.append(get_data('cifar-10-batches-py/data_batch_3'))
batch.append(get_data('cifar-10-batches-py/data_batch_4'))
batch.append(get_data('cifar-10-batches-py/data_batch_5'))

tests = get_data('cifar-10-batches-py/test_batch')
nets = []
lr = 0.1
neuralNet = Net()

for k in range(3):
    for l in range(len(batch)):
        neuralNet.train(batch[l][0], batch[l][1], 1, lr)
        print(10000, l)
    print(k)
    nets.append( (neuralNet.test(tests[0], tests[1]), neuralNet, lr) )

print(nets)
pickle.dump(nets,fRes1)
