# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 12:08:04 2018

@author: Holtech
"""

###################################################################
#           MODELO DE REGRESSÃO LOGÍSTICA USANDO THEANO           #
###################################################################

#Enunciado
# Neste exemplo geramos dados aleatóriamente (toy dataset), fixamos
# um modelo de regressão logística e computamos a acurácia antes e
# depois do treinamento 

#Input data
# Consiste em 1000 vetores com dimensionalidade 100
# Consiste em 1000 amostras com 100 caraterísticas

#Output data
# Consiste em valores 0 e 1 (classes) geradas aleatóriamente

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
import sklearn.metrics

#definir dataset
num_examples = 1000
num_features = 100
dataset = (np.random.randn(num_examples, num_features), 
           np.random.randint(low = 0, high = 2, size = num_examples))

#definir variáveis    
x = T.dmatrix('x')
y = T.dvector('y')
w = shared(np.random.randn(num_features), name = 'w')
b = shared(0., name = 'b')

#definir modelo de regressão logística
#função: y = 1/(1 + exp(x))
net = T.dot(x, w) + b
f_net = 1/(1 + T.exp(-net))

#definir função de erro
#função: erro = -SUM{y * log(y_pred) + (1-y) * log(1 - y_pred)}
error = T.nnet.binary_crossentropy(f_net, y)

#definir função de perda com regularização
#função: -1/N * SUM{y * log(y_pred) + (1-y) * log(1 - y_pred)} + lambda * w^2
def l2(x):
    return T.sum(x**2)

lambda_ = 0.01
loss = error.mean() + lambda_ * l2(w)

#definir cálculo de gradiente
delta_w, delta_b = T.grad(loss, [w, b])

#fase de treinamento => obter w' e b' que otimizam loss
eta = 0.1
train = function(inputs = [x, y], outputs = [f_net, error], 
                 updates = ((w, w - eta * delta_w), (b, b - eta * delta_b)))

#fase de predição
y_pred = f_net > 0.5
predict = function(inputs = [x], outputs = y_pred)

#resultados antes do treinamento
print('Result. pre-treino: ', 
      100 * sklearn.metrics.accuracy_score(dataset[1], predict(dataset[0])), '%')

#resultados depois do treinamento
n_iter = 1000

for i in range(n_iter):
    y_pred, error = train(dataset[0], dataset[1])

#resultados depois do treinamento
print('Result. pós-treino: ', 
      100 * sklearn.metrics.accuracy_score(dataset[1], predict(dataset[0])), '%')
