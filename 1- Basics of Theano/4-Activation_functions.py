# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:42:59 2018

@author: Holtech
"""

######################################################################
#                        FUNÇÕES DE ATIVAÇÃO                         #
######################################################################

#importar pacotes
import theano.tensor as T
from theano import function

#                      função sigmoide: y = 1/(1 + exp(-net))

#definir variável
a = T.dmatrix('a')

#definir função
exp_sigmoid = T.nnet.sigmoid(a)
f_sigmoid = function([a], [exp_sigmoid])

#definir entrada
x_sig = [-1, 0, 1]

#definir saída
y_sig = f_sigmoid([x_sig])

#imprimir resultado
print('\n Sigmoide: ', y_sig)

#       função tangente hiperbólico:  y = (1 - exp(-2x)) / (1 + exp(-2x))

#definir variável
b = T.dmatrix('b')

#definir função
exp_tanh = T.tanh(b)
f_tanh = function([b], [exp_tanh])

#definir entrada
x_tanh = [-1, 0, 1]

#definir saída
y_tanh = f_tanh([x_tanh])

#imprimir resultado
print('\n Tanh: ', y_tanh)

#                          função ReLu: y = max(0,x)

#definir variável
c = T.dmatrix('c')

#definir função
exp_relu = T.nnet.relu(c)
f_relu = function([c], [exp_relu])

#definir entrada
x_relu = [-1, 0, 1]

#definir saída
y_relu = f_relu([x_relu])

#imprimir resultado
print('ReLU: ', y_relu)

#                      função softplus: y = log(1 + exp(x))

#definir variável
d = T.dmatrix('d')

#definir função
exp_splus = T.nnet.softplus(d)
f_splus = function([d], [exp_splus])

#definir entrada
x_splus = [-1, 0, 1]

#definir saída
y_splus = f_splus([x_splus])

#imprimir resultado
print('Softplus: ', y_splus)

#                função softmax: yi = exp(net_i) / SUM{exp(nets)}

#definir variável
e = T.dmatrix('e')

#definir função
exp_smax = T.nnet.softmax(e)
f_smax = function([e], [exp_smax])

#definir entrada
x_smax = [-1, 0, 1]

#definir saída
y_smax = f_smax([x_smax])

#imprimir resultados
print('Softmax: ', y_smax)
