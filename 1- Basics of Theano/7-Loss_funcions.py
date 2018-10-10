# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:51:18 2018

@author: Holtech
"""

###################################################################
#                          FUNÇÕES DE PERDA                       #
###################################################################

import theano.tensor as T
from theano import function

#                       --- binary cross entropy ---
#função: J(w, theta) = -SUM{yi * log(y_pred) + (1 - yi) * log(1 - y_pred)}

#definir variáveis
a1 = T.dmatrix('a1')
a2 = T.dmatrix('a2')

#definir função
exp_a = T.nnet.binary_crossentropy(a1, a2).mean()
f_cost = function([a1, a2], [exp_a])

#definir entradas
x1 = [0.01, 0.01, 0.01]
x2 = [0.99, 0.99, 0.01]

#definir saída
y_bce = f_cost([x1], [x2])

#imprimir resultados
print('Binary cross-entropy: ', y_bce)

#                           --- squared error ---
#função: E = SUM{1/2 * (yi - y_pred)^2}

#definir variáveis
b1 = T.dmatrix('b1')
b2 = T.dmatrix('b2')

#definir função
def squared_error(x, y):
    return T.sqrt(T.sum((x - y) ** 2))

exp_b = squared_error(b1, b2)
f_cost2 = function([b1, b2], [exp_b])

#definir entradas
x1 = [0.01, 0.01, 0.01]
x2 = [0.99, 0.99, 0.01]

#definir saída
y_mse = f_cost2([x1], [x2])

#imprimir resultados
print('Squared error: ', y_mse)

