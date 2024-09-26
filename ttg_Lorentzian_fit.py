# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:23:20 2024

@author: mnk36

Lorentzian fit to BLS peaks
"""
import PyThat
from PyThat import MeasurementTree
import h5py
# %matplotlib qt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

import scipy.optimize as opt


path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp2_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc2p5V_2um.h5'

mt = MeasurementTree(path)
d = mt.dataset
print(d)


#%% Lorentzian fit

# def Lorentz(x, a, b):
    
#      return (1/np.pi * (0.5*a) / ((x-b)**2+(0.5*a)**2)) 
 
    
def Lorentz(x, x0, w, A, y0):
     
     return y0 + (2*A/np.pi * w / (4*(x-x0)**2+(w)**2)) 


#%% read out data

f_RF = d.Frequency_1.values[:,None]
da = d['Acquire spectrum']
ds = d['Get Data']

#%% 

numfrq = d.Frequency.segments  
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
time_bin = 20

frq_offset = np.array([-0.5])[:,None] #bls spectrum offset to RF frequ.

blsF = f_RF+frq_offset 

f_RF_in = [] 
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[a])))
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])))

frq_in = min(range(len(frq)), key=lambda i: abs(abs(frq[i])-f_RF)) # finding the index closest to f_RF


#%% fit

win = 9 # steps in FSR

BLS_frq = frq[(f_bls_in[0]-win):(f_bls_in[0]+win),0]
BLS_frq_window = bls_avg[pos,time_bin,(f_bls_in[0]-win):(f_bls_in[0]+win)]


popt, pcov = opt.curve_fit(Lorentz, BLS_frq, BLS_frq_window)

xdata = np.linspace(np.min(BLS_frq), np.max(BLS_frq), 500) # interpolation
# a, b = popt
# ydata = Lorentz(xdata, a, b)
# ydata = Lorentz(xdata, 0.01, 6.85)

#%% plot one slice

plt.figure(num=20)
plt.clf()
plt.plot(frq[0:frq.size-1,:],bls_avg[pos,time_bin,:], marker='o')
plt.plot(BLS_frq, Lorentz(BLS_frq, *popt), marker='o');

# plt.plot(xdata, ydata, marker='o');
plt.xlabel('Position', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.xlim(np.min(BLS_frq),np.max(BLS_frq))
plt.ylim(-0.1,50)
plt.legend(['Time bin = '+str(time_bin), 'Lorentzian fit'])









