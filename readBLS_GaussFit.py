# reading  bls files with Thatec package

import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
from matplotlib import pyplot as plt
import decay_gauss_convolution as dgc

#%% 

# path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\100nm\cell9_100nm_15GHz_2A_10dBm_10mW_1p5um.h5'
# path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\100nm\cell9_100nm_15GHz_2A_10dBm_10mW_2um.h5'
path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\100nm\cell9_100nm_7GHz_2A_m10dBm_10mW_2um.h5'
# path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\100nm\cell9_100nm_2A_10dBm_10mW_10p5and20p5.h5'



mt = MeasurementTree(path)
d = mt.dataset
print(d)

da = d['Acquire spectrum']

#%%

numfrq = d.Frequency.segments 
frq = d.Frequency.values
loc = d.ScanDimension_1.values 
bls_data = da.values


#%% choosing data

pos = 1
time_bin = 30
f_RF = -7

frq_in = min(range(len(frq)), key=lambda i: abs(abs(frq[i])-f_RF))-1 # finding the index closest to f_RF



#%% parameters

wstrip = 0.1
s = 0.025
x1 = np.arange(-0.5, 1.5+s, s)
x1 = np.round(x1,12) # 0 is not 0
# x1 = np.linspace(-0.5, 1.5,81)
A = 1
B = A*1.1
fwhm = np.array([0.25, 0.5, 0.75, 1, 2])
ld = 0.04

#%% Calculate Gaussian 


x = (loc-8)*0.05 # measurement started at loc 10 -> == 0
decay = dgc.SWdecay(x1,wstrip,A,B,ld)
[gauss,gauss_decay,Gint] = dgc.Gauss_conv(x1,fwhm,wstrip,A,B,ld)


#%% plot measurement

plt.figure(num=1)
plt.plot(x,bls_data[:,frq_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)

#plt.figure(figsize=(8,5))
plt.figure(num=2)
plt.plot(x,bls_data[:,frq_in]/max(bls_data[:,frq_in]), marker='o')
plt.plot(x1,decay, marker='o') #/max(decay)
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.legend([str(f_RF) + ' GHz'])




plt.figure(num=3)
plt.plot(x,bls_data[:,frq_in]/max(bls_data[:,frq_in]), marker='o')
plt.plot(x1,Gint[2,:], marker='o') #/max(decay)
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.legend([str(f_RF) + ' GHz'])






#%% closing the file
hFile = h5py.File(path)
hFile.close()

