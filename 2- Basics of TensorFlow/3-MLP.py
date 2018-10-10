# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:32:40 2018

@author: Holtech
"""

##################################################################
#           REDE NEURAL MLP - PROBLEMA XOR EM TENSORFLOW         #
##################################################################

#importar pacotes
import tensorflow as tf

#definir dataset
#    x1   x2   y
#[1] 0    0    0
#[2] 0    1    1
#[3] 1    0    1
#[4] 1    1    1

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

#criar placeholders para os inputs e outputs
x = tf.placeholder(tf.float32, shape = [4, 2], name = 'x')
y = tf.placeholder(tf.float32, shape = [4, 1], name = 'y')

#definir arquitetura da rede
#   input                   hidden                      output 
#   layer                   layer                       layer
#   x1       Wh[11, 21]     fnet_h[1]      Wo[11]      fnet_o[1]  -> [0 or 1]
#   x2       Wh[12, 22]     fnet_h[2]      Wo[21]      
#  shape = 2                shape = 2                  shape = 1

#hidden layer
Wh = tf.Variable(tf.random_uniform([2, 2], -1, 1), name = 'Wh')
theta_h = tf.Variable(tf.zeros(shape = [2]), name = 'theta_h')

#output layer
Wo = tf.Variable(tf.random_uniform([2, 1], -1, 1), name = 'Wo')
theta_o = tf.Variable(tf.zeros(shape = [1]), name = 'theta_o')

#definição do modelo MLP
net_h = tf.matmul(x, Wh) + theta_h
fnet_h = tf.nn.sigmoid(net_h)
net_o = tf.matmul(fnet_h, Wo) + theta_o
fnet_o = tf.nn.sigmoid(net_o)

#definição de função de custo
#função: J(w, theta) = -1/N * SUM{yi * log(yi_pred) + (1 - yi) * log(1 - yi_pred)}
cost = tf.reduce_mean(-1 * (y * tf.log(fnet_o) + (1 - y) * tf.log(1 - fnet_o)))

#inicializar variáveis
init = tf.initialize_all_variables()

#fase de treinamento
session = tf.Session()
eta = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate = eta).minimize(cost)

list_cost = []
epochs = 100000

session.run(init)
for i in range(epochs):
    session.run(train, feed_dict = {x: X, y: Y})
    list_cost.append(session.run(cost, feed_dict = {x: X, y: Y}))

#resultados
print('Resultados: \n', session.run(fnet_o, feed_dict = {x: X, y: Y}))

'''Resultados: 
0 0 => [[0.01819334]
0 1 => [0.985823  ]
1 0 => [0.98566693]
1 1 => [0.01581946]]'''

#gráfica de resultados - brenchmark
import matplotlib.pyplot as plt
plt.title('MLP para problema XOR - gráfica de custo')
plt.xlabel('#epochs')
plt.ylabel('Cost')           
plt.plot(list_cost)
plt.show()        
