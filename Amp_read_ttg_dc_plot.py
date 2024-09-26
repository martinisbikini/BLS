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

plt.rc('font', size=14)


# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp2_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc3p5V.h5'


#%% time shift

# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc4V.h5'
# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc4V_m40ns.h5'


#%% Amplitude 

# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\.h5'
path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc1p5V.h5'
# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc3V.h5'
# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc4V.h5'
path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\Amp3um_ttg_7p5GHz_3A_0dBm_12mW_dc5V.h5'

#%%% field reversal

# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_400nm\4um\right_high_left_low\Amp4um_7p5GHz_0dBm_m3A_12mW_ttg_2MHz_5V_100ns.h5'
# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_400nm\4um\right_high_left_low\Amp4um_7p5GHz_0dBm_3A_12mW_ttg_2MHz_5V_100ns.h5'


# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_400nm\4um\left_high_right_low\Amp4um_7p5GHz_0dBm_3A_12mW_ttg_2MHz_5V_100ns.h5'
# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_400nm\4um\left_high_right_low\Amp4um_7p5GHz_0dBm_m3A_12mW_ttg_2MHz_5V_100ns.h5'

# path = r'Y:\Martina Kiechle\BLS\SH_amplifiers\Amp_Chip1_600nm\3um\Osc removed\.h5'

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
bls = ds.values

#%%  adding up the reps

bls_avg = np.squeeze(bls, axis=0)#/numreps
# bls_avg = np.sum(bls, axis=0)#/numreps

#%% choosing data

pos = 0
time_bin = 10

frq_offset = np.array([-0.6])[:,None] #bls spectrum offset to RF frequ.

blsF = f_RF+frq_offset 

f_RF_in = [] 
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[a])))
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])))

frq_in = min(range(len(frq)), key=lambda i: abs(abs(frq[i])-f_RF)) # finding the index closest to f_RF

leg = [str(item)+' GHz' for item in f_RF[:,0]]

#%% plot one slice

plt.figure(num=20)
plt.clf()
plt.plot(frq[0:frq.size-1,:],bls_avg[pos,time_bin,:], marker='o')
plt.xlabel('Position', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.ylim(-0.1,100)
plt.legend(['Time bin = '+str(time_bin)])
plt.legend(leg)


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

plot_ttg_frq = np.sum(bls_avg[pos,:,(f_bls_in[0]-8):(f_bls_in[0]+8)],axis=1)[:,None]


fig = plt.figure(num=23, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)

plt.plot(time,np.transpose(bls_avg[pos,:,f_bls_in]), marker='o')
# plt.plot(time,plot_ttg_frq/np.mean(plot_ttg_frq[3:8,:]), marker='o') # normalized cts

left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])

plt.xlabel('Time (ns)', fontsize=16)
plt.ylabel('BLS counts', fontsize=16)
# plt.ylabel('BLS counts (normalized)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.legend(['$V_{DC}$ = 1.5 V', '$V_{DC}$ = 5 V'])
# plt.legend(['$\Delta$t ($V_{DC}$) = 0', '$\Delta$t ($V_{DC}$) = 40 ns'],fontsize=14)
plt.legend(['$\mu_0H_{ext}$ = + 42.1 mT', '$\mu_0H_{ext}$ = - 42.1 mT'],fontsize=14)
# plt.axhline(y=1, xmin=0.0, xmax=1.0, color='black')

#%% closing the file
hFile = h5py.File(path)
hFile.close()