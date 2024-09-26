# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:10:13 2023

@author: mnk36
"""

import PyThat
from PyThat import MeasurementTree
import h5py
# %matplotlib qt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp2_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc2p5V_2um.h5'

mt = MeasurementTree(path)
d = mt.dataset
print(d)

#%% read out data

f_RF = d.Frequency_1.values[:,None]
# frq = d['Frequency']
# dim = d['ScanDimension_1']
da = d['Acquire spectrum']
ds = d['Get Data']

#%% 

numfrq = d.Frequency.segments  
# numreps = d.repetitions.segments
numtim = d.time.segments

time = d.time.values[:,None]
frq = d.Frequency.values[:,None]
loc = d.ScanDimension_1_1.values[:,None]
bls = ds.values

# positioning
numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(numpoin-1)

y_vec = (loc-np.min(loc))*y_step
y_vec = np.round(y_vec,12) # 0 is not 0

#%%  adding up the reps

bls_avg = np.squeeze(bls, axis=0)#/numreps
# bls_avg = np.sum(bls, axis=0)#/numreps

#%% choosing data

pos = 10
time_bin = 10

frq_offset = np.array([-0.5])[:,None] #bls spectrum offset to RF frequ.

blsF = f_RF+frq_offset 

f_RF_in = [] 
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[a])))
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])))

frq_in = min(range(len(frq)), key=lambda i: abs(abs(frq[i])-f_RF)) # finding the index closest to f_RF

#%% plot one slice

plt.figure(num=20)
plt.clf()
plt.plot(frq[0:frq.size-1,:],bls_avg[pos,time_bin,:], marker='o')
plt.xlabel('Position', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.ylim(-0.1,50)
plt.legend(['Time bin = '+str(time_bin)])


#%% plot 2D colormap

## saturated plot

plt.figure(num=21)
plt.clf()
ax = plt.imshow(bls_avg[pos,:,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.min(time),np.max(time)])
plt.colorbar(ax)
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.clim(-0.001,10)

## log plot

# plt.figure(num=22)
# plt.clf()
# ax = plt.imshow(bls_avg[pos,:,:]+1,norm = 'log',aspect ='auto',extent = [np.min(frq),np.max(frq),np.min(time),np.max(time)])
# plt.colorbar(ax)
# plt.xlabel('Frequency (GHz)', fontsize=14)


#%% Time vs. counts

plt.figure(num=23)
plt.clf()
plt.plot(time,np.transpose(bls_avg[pos,:,f_bls_in]), marker='o')
plt.xlabel('Time (ns)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)


#%% plot intensity vs. location 

plot_ttg_frq = np.sum(bls_avg[:,time_bin,(f_bls_in[0]-8):(f_bls_in[0]+8)],axis=1)[:,None]

plt.figure(num=24)
plt.clf()
plt.plot(y_vec,plot_ttg_frq, marker='o')
# plt.plot(y_vec,bls_avg[:,time_bin,f_bls_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
#plt.yscale('log')
plt.tick_params(axis='x',length=6,width=3,labelsize=14)
plt.tick_params(axis='y',which='both',length=6,width=3,labelsize=14)
plt.tight_layout()
plt.legend(['f = '+str(float(f_RF[f_RF_in,0]))+' GHz, Time bin = '+str(time_bin)])


#%% closing the file
hFile = h5py.File(path)
hFile.close()