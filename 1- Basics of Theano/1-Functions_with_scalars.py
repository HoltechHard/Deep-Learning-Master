# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:12:55 2018

@author: Holtech
"""

###########################################################################
#                          FUNÇÕES E ESCALARES                            #
###########################################################################

'''
Considerações para definir funções simples com escalares:
    - escalares são definidos antes de que uma expressão matemática os use
    - cada escalar tem um nome único
    - uma vez definido, o escalar pode ser operado com +, -, *, /
    - as funções construidas x Theano relacionam entradas com saídas
    - podemos computar resultados obtidos x uma função g e suas entradas
      a partir de expressões que não são de Theano
    - gradientes para exemplos triviais e não triviais podem ser computados
      com a mesma facilidade graças à flexibilidade de Theano
'''

#Hands-On

#importar pacotes
import theano.tensor as T
from theano import function

#definição de variáveis escalares
a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')

#definição de uma função
exp = ((a - b + c) * d)/e
f = function([a, b, c, d, e], exp)

#definição de saída apartir de entradas e uma função
output = f(1, 2, 3, 4, 5)

#imprimir resultados
print('\n Result. Esperado: ', ((1 - 2 + 3) * 4)/5)
print('\n Result. Theano: ', output)


