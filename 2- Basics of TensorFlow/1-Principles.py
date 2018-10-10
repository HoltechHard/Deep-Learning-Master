# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:39:27 2018

@author: Holtech
"""

####################################################################
#              PRINCIPIOS FUNDAMENTAIS DO TENSORFLOW               #
####################################################################

# 1) O TENSOR

#Estrutura básica dos dados no Tensorflow, um array multidimensional

import tensorflow as tf

#representação de uma imagem em um tensor
image = tf.image.decode_jpeg(tf.read_file('img/profile.jpg'), channels = 3)
session = tf.InteractiveSession()
print(session.run(tf.shape(image)))
print(session.run(image[0:20, 500:600, 1]))

#fixar um tensor
session = tf.Session()

t1 = tf.zeros(shape = [2, 3, 2])
print(session.run(t1))

t2 = tf.ones(shape = [2, 3, 5])
print(session.run(t2))

t3 = tf.fill(dims = [2, 4], value = 1.5)
print(session.run(t3))

t4 = tf.diag([1, -2, 4])
print(session.run(t4))

t5 = tf.constant([3, 4, 5])
print(session.run(t5))

#fixar sequências de tensores
v1 = tf.range(start = 1, limit = 20, delta = 2)
print(session.run(v1))

#fixar interpolação de tensores
v2 = tf.lin_space(start = 100.0, stop = 123.0, num = 5)
print(session.run(v2))

#fixar um tensor gerado x números aleatórios no intervalo [0,1]
r1 = tf.random_uniform(shape = [2, 3])
print(session.run(r1))

r2 = tf.random_normal(shape = [2, 3], mean = 0.0, stddev = 1.0)
print(session.run(r2))

#operações com matrizes
a = tf.random_uniform(shape = [3, 2])
b = tf.random_normal(shape = [2, 4], mean = 1.0, stddev = 1.0)
c = tf.fill(dims = [3, 4], value = 10.0)

print(session.run(tf.matmul(a, b)))
print(session.run(tf.matmul(a, b) + c))

# 2) COMPUTATIONAL GRAPH E SESSION

#Tensorflow trabalha com 2 ações: construir o grafo e executar o grafo

#Construção do grafo: nós (operações) e arestas (tensores)
#Em teoria, o que vc faz ao criar variáveis, placeholders (espaços 
#reservados) e constantes, é setear o grafo do tensorflow.

#Execução do grafo: usa a sessão para executar as operações no grafo
#session => encapsula o controle e os estados do tensorflow in runtime
#ao criar uma sessão x default se cria um grafo para executar as ops
#para executar qualquer operação primeiro deve-se criar a sessão para
#que esta alloque os recursos e armazene os valores atuais das variáveis

import tensorflow as tf
session = tf.Session()
my_graph = tf.Graph()

with my_graph.as_default():
    variable = tf.Variable(30, name = 'navin')
    initialize = tf.global_variables_initializer()

with tf.Session(graph = my_graph) as session:
    session.run(initialize)
    print(session.run(variable))

# 3) CONSTANTES, PLACEHOLDERS E VARIÁVEIS

#A estrutura de dados representativa é o tensor e possui:
#tipo de dado, rank (profundidade) e shape (dimensionalidade)    

#Definição de constantes
import tensorflow as tf
x = tf.constant(12, dtype = 'float32')
session = tf.Session()
print(session.run(x))

#Definição de variáveis e operações
import tensorflow as tf

#exemplo 01
x = tf.constant(10, dtype = 'float32')
y = tf.Variable(x + 100)
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print('Resultado: ', session.run(y))

#exemplo 02
x1 = tf.constant([14, 23, 40, 30])
y1 = tf.Variable(x1*2 + 100)
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print('Resultados: ', session.run(y1))

#Definição de placeholders
import tensorflow as tf

#exemplo 01
x2 = tf.placeholder('float', None)
y2 = x2*10 + 500

with tf.Session() as session:
    results = session.run(y2, feed_dict = {x2 : [0, 5, 15, 25]})
    print(results)

#exemplo 02
x3 = tf.placeholder('float', [None, 4])    
y3 = x3 * 10 + 1

with tf.Session() as session:
    data_x = [[12, 2, 0, -2], [14, 4, 1, 0], [1, 2, 3, 4]]
    results = session.run(y3, feed_dict = {x3: data_x})
    print(results)

# 4) FUNÇÕES DE ATIVAÇÃO

import numpy as np    
import tensorflow as tf  
import matplotlib.pyplot as plt  

x = np.arange(start = -5.0, stop = 5.0, step = 0.1)

#função sigmoidal: y = 1/(1 + exp(-x))
net_sig = tf.nn.sigmoid(x)
print(session.run(net_sig))    
plt.plot(x, session.run(net_sig))

#função tanh: y = (1 - exp(-2x))/(1 + exp(-2x))
net_tanh = tf.nn.tanh(x)
print(session.run(net_tanh))
plt.plot(x, session.run(net_tanh))

#função ReLU: y = max(0, x)
net_relu = tf.nn.relu(x)
print(session.run(net_relu))
plt.plot(x, session.run(net_relu))

#função ELU: y = { x,                    se x>=0
#                  alpha * (exp(x) - 1), se x<0
net_elu = tf.nn.elu(x)
print(session.run(net_elu))
plt.plot(x, session.run(net_elu))

#função softplus: y = log(1 + exp(x))
net_splus = tf.nn.softplus(x)
print(session.run(net_splus))
plt.plot(x, session.run(net_splus))

#gráfica integrada
plt.figure(1)
plt.plot(x, session.run(net_sig), label = 'sigmoidal')
plt.plot(x, session.run(net_tanh), label = 'tanh')
plt.plot(x, session.run(net_relu), label = 'relu')
plt.plot(x, session.run(net_elu), label = 'elu')
plt.plot(x, session.run(net_splus), label = 'softplus')
plt.title('Activation functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 5) MÉTRICAS

#cálculo da acurácia
import tensorflow as tf

x = tf.placeholder(tf.int32, None)
y = tf.placeholder(tf.int32, None)

acc, acc_op = tf.metrics.accuracy(labels = x, predictions = y)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

vals = session.run([acc, acc_op], feed_dict = {x: [1, 1, 1, 0, 0], 
                                               y: [0, 1, 1, 0, 0]})
accuracy = session.run(acc)
print('Accuracy: ', accuracy)
