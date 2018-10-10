# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:14:26 2018

@author: Hotech
"""

#########################################################
#          1- PRINCIPIOS BÁSICOS DO PYTORCH             #
#########################################################

#importar pacotes
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Preparação, tipos de dado ---

#Tensor 0-D: Escalares
x0 = torch.rand(10)
print(x0.size())
'''Saída: torch.Size([10])'''

#Tensor 1-D: Vetores
x1 = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5])
print(x1.size())
'''torch.Size([5])'''

#Tensor 2-D: Matrizes
from sklearn.datasets import load_boston
data = load_boston()
x2 = torch.from_numpy(data.data)
print(x2.size())
'''Saída: torch.Size([506, 13])'''

#Tensor 3-D: Imagem

'''Instalação do módulo pil: 
   > conda install --channel conda-forge pillow=5 '''
from PIL import Image
panda = np.array(Image.open('img/panda.jpg').resize((224, 224)))
panda_tensor = torch.from_numpy(panda)
print(panda_tensor.size())
'''Saída: torch.Size([224, 224, 3]) '''

#mostrar imagem
plt.imshow(panda)

#mostrar o 1° canal RGB
plt.imshow(panda_tensor[:, :, 0].numpy())

#mostrar um pedaco da imagem
plt.imshow(panda_tensor[50:150, 60:120, 0].numpy())

#Tensor 4-D: Conjunto/Batch de Imagens
from PIL import Image
import glob

#leitura de arquivos do disco
data_path = 'img/cats/'
cats = glob.glob(data_path + '*.jpg')

#converter imagens a vetores numpy
cat_imgs = np.array([np.array(Image.open(cat).resize((64, 64))) for cat in cats])
cat_imgs = cat_imgs.reshape(-1, 64, 64, 3)

#converter vetor multidimensional numpy para tensor
cat_tensors = torch.from_numpy(cat_imgs)
print(cat_tensors.size())
'''Saída: torch.Size([1000, 64, 64, 3])'''

# --- Variáveis ---
from torch.autograd import Variable

#declaracao de variáveis
x1 = Variable(torch.ones(2, 2), requires_grad = True)
y1 = x1.mean() 

#funcao para calcular os gradientes
y1.backward()

#valores associados ao tensor
x1.data

#gradientes 
x1.grad

#funcao gradiente
y1.grad_fn


