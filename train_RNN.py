#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3 Q2 - RNN - Train_RNN
# Submitted by: (Group-7) Bhupesh Dod (21046099), Vinayak G Panchal (21009601), Vinamra Singh (20990294)

# #### Importing Libraries

import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM

# ### Loading the dataset

# data = pd.read_csv("q2_dataset.csv")

# ### Creating the dataset
# Creating the data with 4*3 = 12 fearures, such that for each row in the dataset, the data consists of the last 3 days, features ["Volume", "Open", "High" and "Low"] and the "Open" price of the 4th day as the target variable.

# dataset = list()
# days = 3
# for i in range(len(data) - days):
#     data_features = data[i : i+days+1]
#     features = list()

#     for j in range(days):
#         features.append(data_features.loc[i+days-j][' Volume'])
#         features.append(data_features.loc[i+days-j][' Open'])
#         features.append(data_features.loc[i+days-j][' High'])
#         features.append(data_features.loc[i+days-j][' Low'])
    
#     target = data_features.loc[i][' Open']
#     features.append(target)
#     dataset.append(features)
# dataset = np.array(dataset)

# ### Randmoizing and spliting the dataset into 70% training and 30% test

# np.random.shuffle(dataset)

# train_size = int(len(dataset)*0.7)
# test_size = len(dataset) - train_size

# train_dataset = dataset[:train_size]
# test_dataset = dataset[train_size:]


# ### Saving the created dataset

# pd.DataFrame(train_dataset).to_csv("data/train_data_RNN.csv",index = False, header=False)

# pd.DataFrame(test_dataset).to_csv("data/test_data_RNN.csv",index = False, header=False)


# ### Loading the dataset
# Importing the train datatset (train_data_RNN) from data directory for training the model

train_data = np.loadtxt('data/train_data_RNN.csv',delimiter=',',skiprows=0)
x_train = train_data[:, :-1]
y_train = train_data[:,-1]

# ### Data Pre-processing
# Normalizing the X_train using Standard Scaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# ## Creating the model

model = Sequential()

# Input layer
model.add(LSTM(units = 48, return_sequences = True, input_shape = (x_train.shape[1],1)))

# 1st Hidden layer
model.add(LSTM(units = 82, return_sequences = False))

#Output layer
model.add(Dense(units = 1, activation='linear'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ### Training the Model

model_history = model.fit(x_train, y_train, validation_split = 0.2, batch_size = 8, epochs = 155)

# ### SAVING THE MODEL

model.save("models/21046099_RNN_model.model")

plt.figure(figsize=(20,10))
plt.plot(model_history.history['loss'],label="Train Loss", c= 'r', marker = '.')
plt.plot(model_history.history['val_loss'],label="Validation Loss", c= 'b', marker = '.')
plt.xlabel('Epoch',fontsize=14)
plt.ylabel('Loss (Mean Squared Error)', fontsize=14)
 
plt.legend()
plt.show()