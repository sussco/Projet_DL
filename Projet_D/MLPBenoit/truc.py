# -*- coding: utf-8 -*-
#nécessaire pour pouvoir faire des comms avec des charactères non ascii lolilol
#Pourquoi mxnet ? https://medium.com/@julsimon/keras-shoot-out-tensorflow-vs-mxnet-51ae2b30a9c0


import mxnet as mx
mnist = mx.test_utils.get_mnist()
# permet de charger MNIST en entier dans la mémoire. Possible de l'utiliser aussi pour EMNIST
# les images sont stockées sous un tableau en 4D de la forme
# (taille, 1, 28, 28)
# 1 car une seule chanel de couleurs (gris)
# 28 car taille des images = 28*28

#taille de l'échantillon sur laquelle on va faire les tests
batch_size = 100
#on charge les échantillons de training
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
#on charge les échantillons de test pour voir si notre algo marche
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

#On définie le MLP (multy layer perceptron) en utilisant l'interface symbolique de MXNet
#En gros avec un MLP, il faut réduire notre image de 28*28 en un tableau 1D (c'est pas l'idéal mais ça marche bien, le mieux ce serait de prendre un CNN pour plus de fiabilité)
data = mx.sym.var('data') #on charge notre échantillon dans data
data = mx.sym.flatten(data=data) #on applatit nos images, data devient un tableau 2D (taille_echantillon, nombre-channels*largeur * hauteur) = (100, 784) ici

#une couche plennement connectée (fully connected : FC) demande à ce qu'on face une transformation linéaire de la matrice X de taille n*m (couche précédente) vers la matrice Y de taille n*k
#se calcule avec Y = X W^T + b
#avec
#W la matrice de taille k*m avec les coeffs de passage
#b le bias vector de taille 1*k

#on choisit une ReLU (rectified linear unit) comme fonction d'activation, mais aussi possible de le faire avec toute fonction bijective de R dans [-1,1] comme sigmoid et tanh ou avec des exponentielles tronquées

#######
#on crée la première couche et sa fonction d'activation correspondante
#######
fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type="relu")

#deuxième couche
fc2 = mx.sym.FullyConnected(data=data, num_hidden=64) #on réduit le nombre de params
act2 = mx.sym.Activation(data=fc2, act_type="relu")

#######
#on crée la couche de sortie de taille 10 (car nombre de classes)
#######

fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
#Cette fois pas de ReLU pour la fonction d'activation mais softmax
mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

############### TRAINING ###############
#Vu que notre réseau de neuronnes est crée, on peut commencer à le train
#on utilise la fonction 'module' de MXNet du à son haut niveau d'abstraction
#on a quand même des paramètres à régler, tel que le nombre de paramètres qui vont controller comment l'apprentissage se déroule
#on initialise un module pour entrainer le MLP avec une descente de gradient stochastique(SGD)
#pour que ça pulse, on utilise la fontion 'mini-batch SGD' (cf https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/ ) car plus rapide
#principe du SGD : permet de trouver un juste milieux entre la robustesse de la SGD et l'efficacité / rapidité du 'batch gradient descent'. C'est la méthode la plus utilisée de nos jours

#ici, la taille de notre échantillon est de 100
#il faut aussi que l'on définisee la 'learning rate' (la vitesse à laquelle un réseau abandonne ses anciennes connaissance)
#ici, on prends 0.1 (choix arbitraire mais assez classique)

#ici on fait 10 "tours" (ie : le nombre de fois qu'on injecte notre échantillon dans le réseau), on peut augmenter ce nombre pour obtenir de meilleurs résultats

import logging
logging.getLogger().setLevel(logging.DEBUG)
#on crée un module de training en précisant qu'on souhaite utiliser le CPU (j'ai pas envie de détruire ma CG..) (TODO: modifier ce param pour plus tard car CG = plus rapide)
mlp_model=mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd', #on utilise la méthode SGD pour l'entrainer
                optimizer_params={'learning_rate':0.1}, #on précise la learning rate
                eval_metric='acc', #on précise comment on souhaite évaluer la qualité du MLP
                batch_end_callback=mx.callback.Speedometer(batch_size, 100), #on log le progrès de chaque tours
                num_epoch=30) #on précise le nombre de passage que l'on souhaite effectuer

###PREDICTION
test_iter=mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob=mlp_model.predict(test_iter)
assert prob.shape==(10000,10) #on vérifie que la taille de la prédiction est cohérente

##la métrique utilisée
test_iter=mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
#prédiction de la fiabilité de notre MLP
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.4 #on vérifie que notre réseau est meilleur qu'on enfant de 2 ans
