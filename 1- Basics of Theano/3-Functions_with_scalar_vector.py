# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:13:59 2018

@author: Holtech
"""

#########################################################################
#                     FUNÇÕES, ESCALARES E VETORES                      #
#########################################################################

'''
Considerações para trabalhar com funções simples em escalares e vetores:
    - escalares, vetores e matrizes podem ser usados juntos em expressões
    - tomar em consideração a dimensionalidade dos vetores e matrizes
      enquanto se definam as expressões e passem as entradas
'''

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import function

#definir escalares
a1 = T.dscalar('a1')
a2 = T.dscalar('a2')
a3 = T.dscalar('a3')
a4 = T.dscalar('a4')
a5 = T.dscalar('a5')

#definir matrizes
m1 = T.dmatrix('m1')
m2 = T.dmatrix('m2')
m3 = T.dmatrix('m3')
m4 = T.dmatrix('m4')

#definir função
exp = (((m1 * a1) + (m2  - a2) - (m3 + a3)) * m4 / a4) * a5
f = function([m1, m2, m3, m4, a1, a2, a3, a4, a5], exp)

#definir entradas
x1 = np.array([[1, 1], [1, 1]])
x2 = np.array([[2, 2], [2, 2]])
x3 = np.array([[5, 5], [5, 5]])
x4 = np.array([[3, 3], [3, 3]])

#definir saída
y = f(x1, x2, x3, x4, 1, 2, 3, 4, 5)

#imprimir resultados
print('\n Result. Esperado: ',
      (((x1 * 1.0) + (x2  - 2.0) - (x3 + 3.0)) * x4 / 4.0) * 5.0)
print('\n Result. Theano: ', y)

