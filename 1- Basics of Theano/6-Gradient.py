# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:00:34 2018

@author: Holtech
"""

#################################################################
#                       CÁLCULO DE GRADIENTES                   #
#################################################################

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared    

#definir variáveis
x = T.dmatrix('x')
y = shared(np.array([[4, 5, 6]]))

#definir função
z = T.sum(((x * x) + y) * x)
f = function(inputs = [x], outputs = [z])
#função fica definida: z = SUM{x^3 + xy}

#definir gradiente
exp_grad = T.grad(z, [x])
g = function([x], exp_grad)
#derivada fica definida: dz/dx = 3*x^2 + y

#resultados
print('Original: ', f([[1, 2, 3]]))
print('Grad. Original: ', g([[1, 2, 3]]))

#modificação da variável compartilhada
y.set_value([[1, 1, 1]])
print('Modificado: ', f([[1, 2, 3]]))
print('Grad. Modificado: ', g([[1, 2, 3]]))
