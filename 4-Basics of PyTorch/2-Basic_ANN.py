# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:21:03 2018

@author: Holtech
"""

#####################################################
#         2- PERCEPTRON BÁSICO EM PYTORCH           #
#####################################################

#importar pacotes
import torch
from torch.autograd import Variable
import numpy as np

#                   --- Etapa de Preprocessamento ---

#função para geração/formatação de dados
def get_data():
    #dados coletados
    x_train = np.asarray([3.3, 4.4, 5.5, 6.7, 6.9, 4.1, 9.8, 6.2, 7.6, 2.2, 
                          7.0, 10.8, 5.3, 8.0, 5.7, 9.3, 3.1])
    y_train = np.asarray([1.7, 2.8, 2.1, 3.2, 1.7, 1.6, 3.4, 2.6, 2.5, 1.2, 
                          2.8, 3.5, 1.6, 2.9, 2.4, 2.9, 1.3])
    #tipo de dado
    dtype = torch.FloatTensor
    
    #transformação para Tensor
    x = Variable(torch.from_numpy(x_train).type(dtype), 
                 requires_grad = False).view(17, 1)    
    y = Variable(torch.from_numpy(y_train).type(dtype), 
                 requires_grad = False)
    return x, y


#               --- Definição de Classe Perceptron ---

class PerceptronTorch(object):
    
    #definição de hiperparâmetros
    def __init__(self, eta, n_epochs):
        self.eta = eta
        self.n_epochs = n_epochs
    
    #criação de parâmetros de aprendizado
    def get_weights(self):
        #geração de pesos aleatórios em distribuição normal ux = 0 e std = 1
        self.w = Variable(torch.randn(1), requires_grad = True)
        self.b = Variable(torch.randn(1), requires_grad = True)        
    
    #modelo para o cálculo da saída da rede neural
    def input_net(self, x):
        net = torch.matmul(x, self.w) + self.b
        return net
    
    #predição de resultados
    def predict(self, x):
        y_pred = self.input_net(x)
        return y_pred
    
    #função de custo          
    def loss_fn(self, y, y_pred):        
        loss = (y - y_pred).pow(2).sum()
        
        for param in [self.w, self.b]:
            if not param.grad is None:
                param.grad.data.zero_()
        loss.backward()
        return loss.data[0]
        
    #adaptação dos pesos
    def optimize(self):
        self.w.data = self.w.data - self.eta * self.w.grad.data
        self.b.data = self.b.data - self.eta * self.b.grad.data
    
    #ajuste de modelo
    def fit(self, x, y):
        self.get_weights()
        self.cost = []
        
        for i in range(self.n_epochs + 1):
            y_pred = self.predict(x)
            loss = self.loss_fn(y, y_pred)            
            if(i%50 == 0):
                print('Epoch ', i, ': loss = ', loss)
            self.cost.append(loss)
            self.optimize()
        return self


# ---Execução das classes---

#geração dos dados
x, y = get_data()

# --- Brenchmark Perceptron ---
import matplotlib.pyplot as plt

etas = [1e-4, 1e-5, 1e-6]

def plot_brenchmark(etas):
    
    fig, ax = plt.subplots(nrows = 1, ncols = len(etas), figsize = (12, 4))
    
    for i in range(len(etas)):
        model = PerceptronTorch(eta = etas[i], n_epochs = 1000)
        model.fit(x, y)
        ax[i].plot(range(1, len(model.cost) + 1), model.cost, marker = 'o')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel('MSE')
        ax[i].set_title('Perceptron - eta = ' + str(etas[i]))
    plt.show()

#plotar brenchmark
plot_brenchmark(etas)    

