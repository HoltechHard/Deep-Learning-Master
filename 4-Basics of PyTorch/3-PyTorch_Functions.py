# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:07:08 2018

@author: Holtech
"""

###################################################
#        3- FUNCIONALIDADES DE PYTORCH            #
###################################################

'''
    Etapas para a construção de um algoritmo de Deep Learning:
    
    1. Construir uma pipeline de dados
    2. Construir uma arquitetura de rede
    3. Avaliar a arquitetura usando uma função de custo
    4. Otimizar os pesos da rede utilizando um algoritmo de otimização
'''

#importar pacote
import torch
from torch.autograd import Variable

#Ativação Linear: 
#São equivalentes as camadas Dense ou fully-connected de uma rede
from torch.nn import Linear

#treinamento do modelo
inputs = Variable(torch.randn(1, 10))
model = Linear(in_features = 10, out_features = 5, bias = True)
model(inputs)

#acesso aos parametros do modelo
model.weight
'''
Parameter containing:
-0.0454 -0.1099 -0.2268  0.1460  0.2052  0.0739  0.0100  0.0775  0.1900  0.0777
-0.0050  0.0604  0.1140  0.0122  0.0858 -0.2036 -0.0360  0.2850  0.3017 -0.2751
 0.1255  0.2075  0.0237 -0.2756  0.0603  0.2373  0.1784 -0.1997 -0.2056  0.2944
-0.1599 -0.2861  0.1549 -0.0243  0.0525  0.1934  0.2537 -0.1841 -0.2578  0.2064
 0.2877 -0.2794  0.2147  0.1344  0.1564  0.2997  0.1829 -0.1021  0.1218  0.0644
[torch.FloatTensor of size 5x10]'''

model.bias
'''
Parameter containing:
 0.2208
 0.2452
-0.1153
 0.1328
-0.0708
[torch.FloatTensor of size 5]'''

#Funções de Ativação não-linear
from torch.nn import ReLU

#exemplo RELU
data = Variable(torch.Tensor([1, 2, -1, -2]))
relu = ReLU()
relu(data)

'''
Variable containing:
 1
 2
 0
 0
[torch.FloatTensor of size 4] '''

#Funções de Custo
import torch
import torch.nn
from torch.autograd import Variable

inputs = Variable(torch.randn(3, 5), requires_grad = True)
loss = torch.nn.MSELoss()
target = Variable(torch.randn(3, 5))
output = loss(inputs, target)
output.backward()

inputs.grad
'''
Variable containing:
-0.2487  0.1696 -0.1312 -0.1860  0.0626
 0.0495 -0.0517  0.2347  0.1471 -0.1847
 0.3128 -0.4305  0.0475 -0.0643  0.2355
[torch.FloatTensor of size 3x5]'''

output
'''
Variable containing:
 2.2775
[torch.FloatTensor of size 1]'''

#algoritmos de otimização
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr = 0.01)
