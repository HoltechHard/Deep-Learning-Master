# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:43:15 2018

@author: Holtech
"""

####################################################################
#                         FUNÇÕES E VETORES                        #
####################################################################

'''
Considerações para definir funções simples com vetores:
    - vetores|matrizes sempre são definidas antes de serem usadas 
    - cada vetor|matriz possui um nome único
    - as dimensões do vetor|matriz não são especificadas
    - uma vez definidos o vetor|matriz, podem ser definidos operações como
      +, -, * e o usuário deve tomar em consideração as dimensionalidades
    - depois, pode-se definir uma função que será uma expressão que relacionará
      as entradas para obter uma saída
    - vetores NumPy podem ser passados como parâmetro para computar uma saída
'''

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import function

#definir variáveis
a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')

#definir função
exp = (a + b - c) * d
f = function([a, b, c, d], exp)

#definir entradas
x1 = np.array([[1, 1], [1, 1]])
x2 = np.array([[2, 2], [2, 2]])
x3 = np.array([[5, 5], [5, 5]])
x4 = np.array([[3, 3], [3, 3]])

#definir saída
y = f(x1, x2, x3, x4)

#imprimir resultados
print('Result. Esperado: ', (x1 + x2 - x3) * x4)
print('Result. Theano: ', y)

