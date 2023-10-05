# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:13:02 2023
-------
we have used Python based FLIMview for decay fitting


@author: Mou Adhikari
"""
import flimview.flim as flim
import flimview.io_utils as io
import flimview.plot_utils as pu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import flimview 
from flimview import datasets
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import flimview.models as models
import h5py
from sdtfile import SdtFile
import os
import cv2

#%%
"""read .std file
---------
modified from actual 'read_sdt_file'

"""
def read_sdt_file(sdtfile, channel=0, xpix=256, ypix=256, tpix=1024):
    """
    Reads a sdtfile and returns the header and a data cube.
    Parameters
    ----------
    sdtfile : str
        Path to SdtFile
    channel : int
    xpix : int
    ypix : int
    tpix : int
    Returns
    -------
    3d ndarray
        Read dataset with shape (xpix, ypix, tpix)
    dict
        Header information
    """
    sdt = SdtFile(sdtfile)
    if np.shape(sdt.data)[0] == 0:
        print("There is an error with this file: {}".format(sdtfile))
    sdt_meta = pd.DataFrame.from_records(sdt.measure_info[0])
    sdt_meta = sdt_meta.append(
        pd.DataFrame.from_records(sdt.measure_info[1]), ignore_index=True
    )
    header = {}
    header["flimview"] = {}
    header["flimview"]["sdt_info"] = sdt.info
    header["flimview"]["filename"] = os.path.basename(sdtfile)
    header["flimview"]["pathname"] = os.path.dirname(sdtfile)
    header["flimview"]["xpix"] = xpix
    header["flimview"]["ypix"] = ypix
    header["flimview"]["tpix"] = tpix
    header["flimview"]["tresolution"] = sdt.times[0][1] / 1e-12
    return np.reshape(sdt.data[channel], (xpix, ypix, tpix)), header

#%%
# read .sdt file
sdtfile = '211202_PosA01_100f.sdt'
data, header = read_sdt_file(sdtfile, 0,256,256,1024)
FC = flim.FlimCube(data, header) #FlimCube class
#%% create mask to remove background
# defining kernel 
def get_circular_kernel(diameter):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel

# masking with otsu's thresholding
df = data.sum(axis =2)
c = 255 / np.log(1 + np.max(df))
log_image = c * (np.log(df + 1))
log_image = log_image.astype(np.uint8)
    
    
ret,thr = cv2.threshold(log_image, 0, 255, cv2.THRESH_OTSU)
    
a = get_circular_kernel(9)
a = a.astype(np.uint8)
closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, a)
closing = np.invert(closing)
closing_new = closing.astype(bool)  #converting mask to bool to use as a mask in flimCube fn.

#%%
# apply mask to the .sdt file
FC.mask_intensity(0, mask = closing_new)
pu.plot2d(FC.intensity)

#%%
# fit the data, the fitting procedure is described in Github FLIMVIEW  'https://github.com/mgckind/flimview'
timesteps, mean_pixel = flim.meanDecay (FC)
timesteps_clean, mean_pixel_clean, max_value, time_shift = flim.cleanCurve (timesteps, mean_pixel, norm = True, threshold = 0.02)
mymodel = models.model1
xf, yf, pfit, pcov, chi2 = flim.fitPixel(timesteps, mean_pixel, mymodel, initial_p = None , bounds = (0, np.inf), norm = True)
plt.plot(timesteps - time_shift, mean_pixel/max_value, '.', label = 'original data')
plt.plot(xf, yf, '.', label = 'fitted data')
plt. plot(xf, mymodel(xf, *pfit))
plt.xlabel('time[ns]', fontsize = 15)
plt.ylabel ('intensity', fontsize = 15)
plt. text (6,0.5, flim.printModel (mymodel, pfit, pcov, chi2, oneliner = False))
plt.legend (loc = 0)

#%%
# define boundary
bounds_lower=[0.0, 0.0, 0.0, 0.0]
bounds_upper=[1, 1., 5., 1.]
#fit the whole cube to create a Ffit object
Ffit = flim.fitCube(FC, mymodel,pfit, bounds =(bounds_lower, bounds_upper),norm=True,threshold=0.02)

#%%
#plot the fiiting data with pixel co-ordinate
pu.plotFit(Ffit, FC, 90,150)