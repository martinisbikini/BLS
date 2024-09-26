# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:10:13 2023

@author: mnk36
"""

import PyThat
from PyThat import MeasurementTree
import h5py

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


path = r'O:\public\Martina Kiechle\BLS\Amp2_600nm\Amp2_5um_600nm_8p25GHz_3dBm_n4A_60k_run2.h5'

mt = MeasurementTree(path)
d = mt.dataset
print(d)

#%% read out data

# f_RF = d['Frequency_1']
# frq = d['Frequency']
# dim = d['ScanDimension_1']
da = d['Acquire spectrum']
ds = d['Get Data']

#%% 

numfrq = d.Frequency.segments 
numpoin = d.ScanDimension_1.segments 
numreps = d.repetitions.segments
numtim = d.time.segments

time = d.time.values
frq = d.Frequency.values
loc = d.ScanDimension_1.values 
bls = ds.values

#%%  adding up the reps

bls_avg = np.sum(bls, axis=1)#/numreps

#%% choosing data

f_RF = 8.4
pos = 1
time_bin = 30

frq_in = min(range(len(frq)), key=lambda i: abs(abs(frq[i])-f_RF)) # finding the index closest to f_RF

#%% plot one slice

plt.figure(figsize=(8,5))
plt.figure
plt.plot(frq,bls_avg[pos,time_bin,:], marker='o')
plt.xlabel('Position', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.ylim(-0.1,10)

#%% plot 2D colormap

## saturated plot

plt.figure(figsize=(8,5))
ax = plt.imshow(bls_avg[pos,:,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.min(time),np.max(time)])
plt.colorbar(ax)
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.clim(-0.001,10)

## log plot

plt.figure(figsize=(8,5))
ax = plt.imshow(bls_avg[pos,:,:]+1,norm = 'log',aspect ='auto',extent = [np.min(frq),np.max(frq),np.min(time),np.max(time)])
plt.colorbar(ax)
plt.xlabel('Frequency (GHz)', fontsize=14)


#%% plot intensity vs. location 

plt.figure(figsize=(8,5))
plt.plot(loc,bls_avg[:,time_bin,frq_in], marker='o')
plt.xlabel('Position', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
#plt.yscale('log')
plt.tick_params(axis='x',length=6,width=3,labelsize=14)
plt.tick_params(axis='y',which='both',length=6,width=3,labelsize=14)
plt.tight_layout()
plt.legend([str(f_RF) + ' GHz'])


#%% closing the file
hFile = h5py.File(path)
hFile.close()