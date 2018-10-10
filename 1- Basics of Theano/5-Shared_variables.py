# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 20:07:38 2018

@author: Holtech
"""

######################################################################
#                       VARIÁVEIS COMPARTILHADAS                     #
######################################################################

'''
Considerar para variáveis compartilhadas:
    - todos os modelos podem envolver funções definidas com estados internos
    - a variável compartilhada é definida usando um construtor de Theano
    - a variável compartilhada pode ser inicializada x construtor numpy
    - a variável compartilhada pode ser usada para definir expressões e fn
    - pode retornar o valor de uma variável compartilhada usando get_value
    - pode insertar valor a uma variável compartilhada usando set_value
    - uma função que ao ser definida usa uma variável compartilhada, gera uma
      saída baseada no valor atual da variável compartilhada 
'''

#importar pacotes
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared

#definir variáveis
x = T.dmatrix('x')
y = shared(np.array([[4, 5, 6]]))
z = x + y   

#definir função
f = function(inputs = [x], outputs = [z])

#imprimir resultados
print('shared value: ', y.get_value())
print('availação na função: ', f([[1, 2, 3]]))

#modificação na variável compartilhada
y.set_value([[5, 6, 7]])

#imprimir novos resultados
print('shared value mod: ', y.get_value())
print('avaliação mod: ', f([[1, 2, 3]]))


