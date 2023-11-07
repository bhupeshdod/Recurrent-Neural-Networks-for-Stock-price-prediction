#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3 Q2 - RNN - Test_RNN
# Submitted by: (Group-7) Bhupesh Dod (21046099), Vinayak G Panchal (21009601), Vinamra Singh (20990294)

# ### Importing the libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from tensorflow.keras.layers import LSTM

# ### Loading the dataset
# Loading test dataset (test_data_RNN) from data directory to evaluate the model

test_data = np.loadtxt('data/test_data_RNN.csv',delimiter=',',skiprows=0)

x_test = test_data[:, :-1]
y_test = test_data[:,-1]

# ### Data Pre-processing
# Normalizing the X_test using Standard Scaler

scaler = StandardScaler()

x_test = scaler.fit_transform(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# ### Loading the model
# Loading the saved model.model file from models directory for testing

model = tf.keras.models.load_model("models/21046099_RNN_model.model")

# ## Predicting the outputs
# Predicting the test stock prices using test dataset and loaded model to evaluate the model

pred = model.predict(x_test)
scores = model.evaluate(x_test,y_test)

plt.figure(figsize=(20,10))
plt.plot(y_test, color="red", marker='o', linestyle='dashed', label="Actual Stock Price")
plt.plot(pred, color="blue", marker='o', linestyle='dashed', label="Predicted Stock Price")
plt.title("Stock Price Prediction",fontsize=16)
plt.xlabel("Date (random)",fontsize=14)
plt.ylabel("Stock Price",fontsize=14)
plt.legend()
plt.show()
