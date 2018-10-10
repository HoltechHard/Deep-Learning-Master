# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:51:36 2018

@author: Holtech
"""

##################################################################
#                  PRINCIPIOS BÁSICOS DO KERAS                   #
##################################################################

#Pasos para criar modelos de deep learning no Keras

# 1) importar dados: 
# dataset de 50'000 32 x 32 imagens de cor com 10 categorías de imagem
# e 10'000 imagens de teste
# classes: airplane | automobile | bird | cat | deer | dog | frog | 
#          horse | ship | truck

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

np.random.seed(100)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2) preprocessar os dados

#Flattening os dados para MLP: 3 x 32 x 32
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

#Standarizacao dos dados X
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
x_train = sc_train.fit_transform(x_train)
sc_test = StandardScaler()
x_test = sc_test.fit_transform(x_test)

#Categorizacao das saídas Y
classes = 10
y_aux = y_test
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

# 3) definir o modelo [camadas: convolution, pooling, dropout, 
#    batch normalization e activation function]

#Arquitetura do modelo: Imagem: [3 x 32 x 32]
#  input layer   => hidden h1 => hidden h2 => output layer
# 3,072 features => 512 nodes => 120 nodes => 10 nodes

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()

#input e hidden layer h1
model.add(Dense(units = 512, input_shape = (3072, )))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.4))

#hidden layer h2
model.add(Dense(units = 120))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.2))

#output layer
model.add(Dense(units = classes))
model.add(Activation('softmax'))

# 4) compilar o modelo. Aplicar um otimizador sob uma funcao de custo
from keras.optimizers import adam
adam = adam(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, 
                  metrics = ['accuracy'])

# 5) fixar o modelo com dados de treinamento
#10 epochs => recorridos ao longo de todas as iteracoes do conjunto de treino
#50 iteracoes => train_set = 50'000 e batch_size = 1'000
model.fit(x_train, y_train, batch_size = 1000, epochs = 10000, 
              validation_data = (x_test, y_test))

# 6) avaliar o modelo 
score = model.evaluate(x_test, y_test, verbose = 0)
print('Acurácia: ', 100 * score[1], '%')

# 7) realizar predicoes usando dados de teste
y_pred = model.predict_classes(x_test)
