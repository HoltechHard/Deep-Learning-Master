# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:17:37 2018

@author: Holtech
"""

###############################################################
#              REDE NEURAL MLP USANDO THEANO                  #
###############################################################

#Enunciado 
# No exemplo se geram dados de forma aleatória e se define uma
# rede neural de 2 camadas que envolve:
# 1° Layer:
# dados de entrada: x, saídas: y, bias: b1, vetor de pesos: w1 
# 2° Layer:
# bias: b2, vetor de pesos: w2
# Os bias e os vetores de pesos são variáveis compartilhadas

#Input data
# Consiste em 1000 amostras de 100 caraterísticas cada uma

#Output data
# Saídas geradas 0 e 1 gerados aleatóriamente

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import shared
from theano import function
import sklearn.metrics

#definir dataset
num_examples = 1000
num_features = 100
hidden = 10
dataset = (np.random.randn(num_examples, num_features), 
           np.random.randint(low = 0, high = 2, size = num_examples))

#definir variáveis
x = T.dmatrix('x')
y = T.dvector('y')
Wh = shared(np.random.randn(num_features, hidden), name = 'Wh')
theta_h = shared(np.zeros(hidden), name = 'theta_h')
Wo = shared(np.random.randn(hidden), name = 'Wo')
theta_o = shared(0., name = 'theta_o')

#definir modelo de rede neural MLP
net_h = T.dot(x, Wh) + theta_h
fnet_h = T.nnet.sigmoid(net_h)
net_o = T.dot(fnet_h, Wo) + theta_o
fnet_o = T.nnet.sigmoid(net_o)

#definir função de erro
#função: error = -SUM{yi * log(yi_pred) + (1 - yi) * log(1 - yi_pred)}
error = T.nnet.binary_crossentropy(fnet_o, y)

#definir função de custo
#função: loss = 1/N * error + lambda * SUM{wi}

def l2(x):
    return T.sum(x**2)

lambda_ = 0.01  
loss = error.mean() +  lambda_ * (l2(Wh) + l2(Wo))

#definir cálculo do gradiente
delta_Wh, delta_thetah, delta_Wo, delta_thetao = T.grad(loss, 
                                                    [Wh, theta_h, Wo, theta_o])

#fase de treinamento => obter Wh', theta_h', Wo' e theta_o' que otimizam loss
eta = 0.1
train = function(inputs = [x, y], outputs = [fnet_o, error], updates = 
            ((Wh, Wh - eta * delta_Wh), (theta_h, theta_h - eta * delta_thetah), 
            (Wo, Wo - eta * delta_Wo), (theta_o, theta_o - eta * delta_thetao)))

#fase de predição 
y_pred = fnet_o > 0.5
predict = function(inputs = [x], outputs = y_pred)

#resultados antes do treinamento
print('Result. pre-treino: ', 
      100 * sklearn.metrics.accuracy_score(dataset[1], predict(dataset[0])), '%')

#resultados depois do treinamento
n_iter = 10000
for i in range(n_iter):
    y_pred, error = train(dataset[0], dataset[1])

print('Resul. pós-treino: ', 
      100 * sklearn.metrics.accuracy_score(dataset[1], predict(dataset[0])), '%')

