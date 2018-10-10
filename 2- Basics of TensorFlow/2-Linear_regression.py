# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:49:51 2018

@author: Holtech
"""

################################################################
#        MODELO DE REGRESSÃO LINEAR USANDO TENSORFLOW          #
################################################################

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston

#importar dataset
def read_infile():
    dataset = load_boston()
    features = dataset.data
    target = dataset.target
    return features, target

#preprocessamento: standarizar dados => x' = (x - ux) / std(x)
def standarize_data(data):
    ux = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    return (data - ux)/std

#adicionar o bias a matriz de caraterísticas
def append_bias(features, target):
    n_rows = features.shape[0]
    n_cols = features.shape[1]
    bias = np.ones(shape = (n_rows, 1))
    x = np.concatenate((features, bias), axis = 1)
    x = np.reshape(x, newshape = [n_rows, n_cols + 1])
    y = np.reshape(target, newshape = [n_rows, 1])
    return x, y

#leitura de dados
features, target = read_infile()
std_features = standarize_data(features)
X, Y = append_bias(std_features, target)

#criação de tensores, variáveis, placeholders
n_cols = X.shape[1]
x = tf.placeholder(tf.float32, shape = [None, n_cols], name = 'x')
y = tf.placeholder(tf.float32, shape = [None, 1], name = 'y')
w = tf.Variable(tf.random_normal(shape = [n_cols, 1]), name = 'weights')
init = tf.global_variables_initializer()    

#fase de treinamento
eta = 0.01
y_pred = tf.matmul(x, w)
error = y - y_pred
cost = tf.reduce_mean(tf.square(error))
train = tf.train.GradientDescentOptimizer(learning_rate = eta).minimize(cost)

#execução do gradiente descendente
list_cost = []
epochs = 1000
with tf.Session() as session:
    session.run(init)
    for i in range(epochs):
        session.run(train, feed_dict = {x: X, y: Y})
        list_cost.append(session.run(cost, feed_dict = {x: X, y: Y}))
    y_pred_ = session.run(y_pred, feed_dict = {x: X, y: Y})
    error_ = session.run(error, feed_dict = {x: X, y: Y})

#resultados
print('Resultados: MSE = ', list_cost[-1])

#visualizar resultados
import matplotlib.pyplot as plt
plt.title('Regressão Linear: Gráfica de custo')
plt.xlabel('#epochs')
plt.ylabel('Cost')
plt.plot(list_cost)
plt.show()    
    
