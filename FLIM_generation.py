# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:56:47 2023
Artificial data generation method which is simial to 'Fluorescence lifetime imaging data' with 'FLIM' function
-------------------------

@author: Mou Adhikari
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
import math
from sklearn.preprocessing import MinMaxScaler

arr = []    # create an empty list to insert the intensity (y) values 
A1 = []     # create an empty list to insert the abundance (a1) values
A2 = []     # create an empty list to insert the abundance (a2) values
     # create an empty list to insert the abundance (a3) values
T1 = []     # create an empty list to insert the lifetime (t1) values
T2 = []     # create an empty list to insert the lifetime (t2) values
Lamda = []  # create an empty list to insert the Poisson noise (lamda) values
N1 = []            
STD = []
only_data = []
Noise = []
# create a function to generate decay traces

def FLIM (n, channels, Ltime, Utime):
    """ 
    n is the number of decay traces
    channels = number of data point 
    Ltime = lower limit
    Utime = upper limit
    
    """
    
    for i in range (n):
        x = np.linspace (Ltime, Utime, num = channels)
        standard_dev = random.randint(1,10)
        g = signal.gaussian (50, std = standard_dev)
        l=random.uniform (1,10)
        a1 = random.uniform (0.0, 100.0)
        a2 = 100 - a1
            
        t1 = (1.0-0.05) * np.random.random_sample()+0.05
        t2 = random.uniform (1.0, 4.0)
        I = ((a1/100)*np.exp(-x/t1) + (a2/100)*np.exp (-x/t2))
        I1=np.convolve(I,g, mode='full')
        I1 = I1[0:channels]
        t=np.linspace(100,110,num=channels)
        I2= np.random.poisson(l/t, channels)
        y = I1 + I2
        y = (y-min(y))/(max(y)- min(y))    
        
        arr.insert(0,y)
        A1.insert(0,a1)
        A2.insert(0,a2)
        T1.insert(0,t1)
        T2.insert(0,t2)
        STD.insert (0,standard_dev )
        N1.insert(0,I)
        only_data.insert(0,I1)
        Lamda.insert (0, l)
        Noise.insert(0, I2)
        fig=plt.figure(i)
        plt.plot(x,y)
        #plt.show(fig)
#%%
"""call function FLIM to generate decay traces"""
 FLIM (5000, 1024, 0, 10)
#%%
#convert all list values into an array
Flim_array=np.array(arr)
Flim_A1=np.array(A1)
Flim_A2=np.array(A2)
Flim_T1=np.array(T1)
Flim_T2=np.array(T2)
#Flim_std=np.array(STD)
Flim_N1=np.array(N1)
Flim_Lamda = np.array(Lamda)
Flim_I1 = np.array(only_data)
Flim_I2 = np.array(Noise)

#%%
#convert array tp panda dataframe
Flim_dataframe=pd.DataFrame(Flim_array)
Flim_A1dataframe = pd.DataFrame(Flim_A1, columns = ['a1'])
Flim_A2dataframe = pd.DataFrame (Flim_A2, columns = ['a2'])
Flim_T1dataframe = pd.DataFrame (Flim_T1, columns = ['t1'])
Flim_T2dataframe = pd.DataFrame (Flim_T2, columns = ['t2'])
Flim_stddataframe=pd.DataFrame(Flim_std, columns = ['std'])
Flim_N1datafram = pd.DataFrame(Flim_N1)
Flim_I2datafram = pd.DataFrame(Flim_I2)
Flim_I1datafram = pd.DataFrame(Flim_I1)
Flim_Lamdadatafram = pd.DataFrame(Flim_Lamda)