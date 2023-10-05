# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:51:46 2023
Two step training process:
    1st step- autoencoder training (denoise data)
    2nd step- CNN training (with the bottle neck feature of the autoencoder trained the CNN to predict the target value (data_y))
-------------------------
This auto-encoder model is trained with 'noise-in-denoise-out' 

@author:Mou Adhikari
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import DepthwiseConv1D
from tensorflow.keras.layers import Dropout, concatenate,Conv1DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tabulate import tabulate
import tensorflow.keras.backend as K
import pickle
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

""" Read artificial data without noise"""
normal_x = pd.read_excel(r"C:\Users\fe73yap\Downloads\Result_spectrum_Without_noise.xlsx")

#%%
""" Read artificial data only noise """
noise_x = pd.read_excel(r"C:\Users\fe73yap\Downloads\Result_spectrum_only_noise.xlsx")

#%%
""" Read target """
data_y = pd.read_excel(r'C:\Users\fe73yap\Downloads\results_spectrum_new_6_16_22_1.xlsx')

#%%%
data_y = data_y.iloc[: , 1:]
data_y_abundances = data_y[['a1','a2']]
data_y_lifetime = data_y [['t1','t2']]
data_y_abundances = data_y_abundances.div(100) # normalize abundances with maximum value
data_y_lifetime_t1 = (data_y_lifetime [['t1']]).div(1.0) #normalize lifetime 1 (t1) with maximum value
data_y_lifetime_t2 = (data_y_lifetime [['t2']]).div(4.0) # normalize lifetime 2 value with maximum value

data_y_lifetime_new = pd.concat([data_y_lifetime_t1,data_y_lifetime_t2,data_y_lifetime_t3 ], axis = 1)
data_Y = pd.concat([data_y_abundances,data_y_lifetime_new], axis = 1) # data_Y is ready for training
#%%
"""train-test-validation only noise split"""
x_train_noise,x_test_noise=train_test_split(noise_x,test_size=0.2,random_state=1)
x_train_noise,x_val_noise = train_test_split(x_train_noise,test_size=0.25,random_state=1)

#%%%
x_train,x_test,y_train,y_test=train_test_split(normal_x,data_Y,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=1)

#%%
"""create the input training data for autoencoder training"""
noisy_x_test = np.add(x_test, x_test_noise)
noisy_x_val = np.add(x_val, x_val_noise)
noisy_x_train = np.add(x_train, x_train_noise)
#%% 
""" Input of encoder"""
#normalize the training, validation and test data
noisy_X_train = preprocessing.normalize(noisy_x_train)
noisy_X_test = preprocessing.normalize(noisy_x_test)
noisy_X_val = preprocessing.normalize(noisy_x_val)

#%%
# only decay trace without noise
"""output of the autoencoder"""
X_train = preprocessing.normalize(x_train) # only decay trace without noise
X_test = preprocessing.normalize(x_test)
X_val = preprocessing.normalize(x_val)

#%%
#define validation data
valid_data = (noisy_X_val,X_val/0.01)
#%%
"""define autoencoder model"""
n_inputs = 1024
# build unet model
inputs = tf.keras.layers.Input((1024,1))
c1 = tf.keras.layers.Conv1D(64,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(inputs)
c1 = BatchNormalization()(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv1D(64,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c1)
c1 = BatchNormalization()(c1)
p1 = tf.keras.layers.MaxPooling1D(pool_size =2)(c1)


c2 = tf.keras.layers.Conv1D(128,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p1)
c2 = BatchNormalization()(c2)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv1D(128,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c2)
c2 = BatchNormalization()(c2)
p2 = tf.keras.layers.MaxPooling1D(pool_size = 2)(c2)

c3 = tf.keras.layers.Conv1D(256,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p2)
c3 = BatchNormalization()(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv1D(256,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c3)
c3 = BatchNormalization()(c3)
p3 = tf.keras.layers.MaxPooling1D(pool_size = 2)(c3)


c4 = tf.keras.layers.Conv1D(512,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(p3)
c4 = BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv1D(512,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(c4)
c4 = BatchNormalization()(c4)


#Expansive path
u6 = Conv1DTranspose(256,2, strides = 2, padding = 'same')(c4)
u6 = concatenate ([u6,c3])
c6 = Conv1D(256,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(0.2)(c6)
c6 = Conv1D(256,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c6)
c6 = BatchNormalization()(c6)

u7 = Conv1DTranspose(128,2, strides = 2, padding = 'same')(c6)
#u7 = concatenate ([u7,c2])
c7 = Conv1D(128,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(0.2)(c7)
c7 = Conv1D(128,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c7)
c7 = BatchNormalization()(c7)


u8 = Conv1DTranspose(64,2, strides = 2, padding = 'same')(c7)
#u8 = concatenate ([u8,c2])
c8 = Conv1D(64,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(0.2)(c8)
c8 = Conv1D(64,7, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c8)
#c8 = Conv1D(64,3, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c8)
#c8 = Conv1D(64,1, activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c8)
c8 = tf.keras.layers.DepthwiseConv1D(1, depth_multiplier = 2, activation = 'relu',depthwise_initializer='he_normal', padding = 'same' )(c8)

c8 = BatchNormalization()(c8)




outputs = Conv1D (1,1, activation = 'LeakyReLU')(c8)

model = Model (inputs = [inputs], outputs = [outputs])
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile (optimizer = opt,loss = 'mse')

#%% Train the model
history = model.fit(noisy_X_train, X_train/0.01, epochs = 1000, batch_size = 32, verbose = 1, validation_data = valid_data)

#%%
"""predict the output of autoencoder"""
y_pred = model.predict(noisy_X_test, verbose=1)

#%%
"""CNN training
-------------
take the bottleneck from autoencoder and feed it as a CNN input
 """
encoder = Model(inputs=inputs, outputs=c4)

#%%
# encode the train data
x_train_encode = encoder.predict(X_train)
# encode the test data
x_test_encode = encoder.predict(X_test)
##reshape for CNN training
x_val_encode = encoder.predict(X_val)
#reshape for CNN training
x_train_encode_new =x_train_encode.reshape(6000,65536,1)
#reshape for CNN training
x_test_encode_new = x_test_encode.reshape(2000,65536,1)
#reshape for CNN training
x_val_encode_new = x_val_encode.reshape(2000,65536,1)

#%%
"""CNN Architecture """
# CNN architecture
model1 = Sequential()
model1.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(65536,1)))
model1.add(MaxPooling1D(pool_size=2))

model1.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))

model1.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model1.add(MaxPooling1D(pool_size=2))

#model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#model.add(MaxPooling1D(pool_size=2))



model1.add(Flatten())
model1.add(Dense(400, activation = 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(6, activation = 'relu'))# change dense layer output everytime
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
#model.compile(optimizer=opt, loss='mse', metrics=['mae'])
model1.compile(optimizer=opt, loss= 'mse', metrics=['mae'])
print(model1.summary())

#%%
"""train CNN model"""
model1.fit(x_train_encode_new, y_train, epochs =50, validation_data = [x_val_encode_new, y_val])
#%%
"""test CNN output with artificial data"""
y_test_pred = model1.predict(np.reshape(x_test_encode_new,[2000,65536,1]))

#%%
"""test CNN model output with experimental data 
----------------------
Experimental  data has 2 channels (R and G)
----------------------
here will test CNN model with R and G data individually

"""
# read R channel data from .csv file
real_data_R =pd.read_excel(r'C:\Users\fe73yap\Downloads\Mean_decaytrace_R and G channel(2).xlsx',usecols ="A:AMM" )

#%%
#predict lifetime parameters with CNN model
real_data_R = real_data_R.to_numpy()
real_data_R =  preprocessing.normalize(real_data_R)
rl_data_R_pad = encoder.predict(real_data_R)
yhat_real_R = model1.predict(np.reshape(rl_data_R_pad,[386,65536,1]))

#%%
# read G channel data from .csv file
real_data_G =pd.read_excel(r'C:\Users\fe73yap\Downloads\Mean_decaytrace_R and G channel(2).xlsx',usecols ="AMO:BZX" )

#%%
#predict lifetime parameters with CNN model
real_data_G = real_data_G.to_numpy()
real_data_G =  preprocessing.normalize(real_data_G)
rl_data_G_pad = encoder.predict(real_data_G)
yhat_real_G = model1.predict(np.reshape(rl_data_G_pad,[386,65536,1]))

