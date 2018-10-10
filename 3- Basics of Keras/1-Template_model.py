# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:41:09 2018

@author: Holtech
"""

############################################################
#              TEMPLATE DE MODELO NO KERAS                 #
############################################################

#importar pacotes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#importar dados: 500 amostras de 100 caraterísticas
dataset = (np.random.random(size = (500, 100)), 
           np.random.randint(low = 0, high = 2, size = (500, 1)))

#preprocessamento: dividir conjunto de treinamento - conjunto de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], 
                                            test_size = 0.2, random_state = 0)

#preprocessamento: standarizacao dos dados
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
x_train = sc_train.fit_transform(x_train)
sc_test = StandardScaler()
x_test = sc_train.fit_transform(x_test)

#definir modelo
model = Sequential()
model.add(Dense(units = 12, input_shape = (100, ), activation = 'relu'))
model.add(Dense(units = 8, activation = 'relu'))    
model.add(Dense(units = 1, activation = 'sigmoid'))

#compilar modelo
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
                  metrics = ['accuracy'])

#fixar o modelo
model.fit(x_train, y_train, batch_size = 20, epochs = 500, 
              validation_data = (x_test, y_test))

#avaliacao do modelo
score = model.evaluate(x_test, y_test, verbose = 0)
print('Acurácia', 100 * score[1], '%')

#realizar predicoes usando dados de teste
y_prob = model.predict(x_test)
y_pred = np.where(y_prob > 0.5, 1, 0)

