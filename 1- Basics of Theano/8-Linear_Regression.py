# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 00:41:09 2018

@author: Holtech
"""

#####################################################################
#               MODELO DE REGRESSÃO LINEAR USANDO THEANO            #
#####################################################################

#Enunciado
# No exemplo se geram alguns dados de forma aleatória (toy dataset) 
# e fixamos um modelo de regressão linear que será computado antes
# e depois do treinamento usando a medida RMSE.

#Input data
# Consiste em 1000 vetores com dimensionalidade 100 
# São 1000 exemplos com 100 caraterísticas

#Output data
# Consiste em valores gerados aleatóriamente

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import shared
from theano import function
import sklearn.metrics

#gerar o toy dataset
num_examples = 1000
num_features = 100
dataset = (np.random.randn(num_examples, num_features), 
           np.random.randn(num_examples))

#definir variáveis 
x = T.dmatrix('x')
y = T.dvector('y')
w = shared(np.random.randn(num_features), name = 'w')
b = shared(0., name = 'b')

#definir modelo linear
net = T.dot(x, w) + b

#definir função de erro
def squared_error(x, y):
    return (x - y)**2

error = squared_error(y, net)
    
#definir função de custo com regularização
#função: loss = min{[y - (x * w + b)]^2} + lambda * w^2
def l2(x):
    return T.sum(x**2)

lambda_ = 0.01
loss = error.mean() + lambda_ * l2(w)   

#definir calculo do gradiente
delta_w, delta_b = T.grad(loss, [w, b])

#fase de treinamento => obter w' e b' que otimizam loss 
eta = 0.1
train = function(inputs = [x, y], outputs = [net, error], 
                 updates = ((w, w - eta * delta_w), (b, b - eta * delta_b)))

#fase de predição => obter y_pred = x * w' + b'
predict = function(inputs = [x], outputs = net)

#resultados antes do treinamento
print('RMSE pre-treino: ', 
      sklearn.metrics.mean_squared_error(dataset[1], predict(dataset[0])))

#resultados depois do treinamento
iter_ = 1000

for i in range(iter_):
    y_pred, error = train(dataset[0], dataset[1])

print('RMSE pós-treino: ', 
      sklearn.metrics.mean_squared_error(dataset[1], predict(dataset[0])))

