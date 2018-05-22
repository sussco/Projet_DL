from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt


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




def generateImageCifar(nb):

    batch = []
    batch.append(get_data('cifar-10-batches-py/data_batch_1'))
    batch.append(get_data('cifar-10-batches-py/data_batch_2'))
    batch.append(get_data('cifar-10-batches-py/data_batch_3'))
    batch.append(get_data('cifar-10-batches-py/data_batch_4'))
    batch.append(get_data('cifar-10-batches-py/data_batch_5'))

    # tests = get_data('cifar-10-batches-py/test_batch')

    trainings = []
    labels = []
    for i in range(len(batch)):
        trainings.append(batch[i][0])
        labels.append(batch[i][1])


    trainings = np.reshape(np.array(trainings), (50000, 3, 32, 32))
    trainingLabels = np.reshape(np.array(labels), (50000,10))


    # test = np.reshape(np.array(tests[:][0]), (10000, 3, 32, 32))
    # testLabels = np.reshape(np.array(tests[:][1]), (10000,10))
    print("done...")



    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest', data_format = "channels_first")


    sample = []
    lab = []
    for btc in datagen.flow(trainings, trainingLabels, batch_size=10):
        i += 1
        sample.append(np.reshape(np.array(btc[0]),(10,3,32,32)).transpose([0, 2, 3, 1]) )
        lab.append(btc[1][0])
        if i > nb+3:
            break
    print("generating images done.")
    return np.array(sample), np.array(lab)



# a = get_data('cifar-10-batches-py/data_batch_1')
# samp = np.reshape(a[0][4], (1,3,32,32))
# sampL = np.reshape(a[0][1], (1,3,32,32))
#
# datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest', data_format = "channels_first")
#
# sample = []
# lab = []
# i = 0
# for btc in datagen.flow(samp, sampL, batch_size=10):
#     i += 1
#     sample.append(np.reshape(np.array(btc[0]),(3,32,32)).transpose([1, 2, 0]) )
#     lab.append(btc[1][0])
#     if i > 10:
#         break
#
#
#
# fig = plt.figure(figsize = (20,20))
#
# for i in range(1,11):
#     fig.add_subplot(1, 10, i)
#     plt.axis('off')
#     print(sample[i-1].shape)
#     plt.imshow(sample[i-1],cmap=plt.cm.gray)
#
# plt.show()
