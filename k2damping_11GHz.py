# read frequency sweep vs. amplitude


import xarray as xr
import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as ptc

import sys
sys.path.append('O:\public\Martina Kiechle\BLS\Data analysis')
import edge_gauss_convolution as dgc

#%% 


path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_11GHz_4A_5dBm_10mW_4um.h5'

mt = MeasurementTree(path)
d = mt.dataset
print(d)


#%%

numfrq = d.Frequency.segments 
numfrf = d.Frequency_1.segments
# numpoin = d.ScanDimension_1.segments 
#numreps = d.repetitions.segments
#numtim = d.time.segments
#time = d.time.values
frq = d.Frequency.values
frq = np.expand_dims(frq, axis=1)
f_RF = d.Frequency_1.values
loc = d.ScanDimension_1_1.values 
bls_data = d['Acquire spectrum'].values
bls_data = bls_data.astype('f')
beam_ref = np.transpose(d['Ch 4'].values)
numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

# dx = d['ROI - relative positioning'].values[0,0]
# dy = d['ROI - relative positioning'].values[0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(80-1)


y_vec = np.expand_dims(loc*y_step,axis=1)
y_vec = np.round(y_vec,12) # 0 is not 0

leg = [str(item)+' GHz' for item in f_RF]


#%% choosing data

pos = 50

frq_offset = [0.25,0.375]

blsF = f_RF-frq_offset #bls spectrum offset = -0.25 GHz atm

f_RF_in = []#np.linspace(0, len(f_RF)-1,len(f_RF))
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[a])))
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])))
    # f_bls_in[a] = min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])) # finding the index closest to f_RF in BLS frequencies


#%% plot one slice

plt.figure(num=100)
plt.clf()
plt.plot(frq,np.transpose(bls_data[:,pos,:]), marker='o')
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
# plt.ylim(-0.1,5e4)
plt.ylim(-0.1,bls_data[f_RF_in[0],pos,f_bls_in[0]]+200)
plt.legend(leg)


plt.figure(num=101)
plt.clf()
plt.plot(frq,np.transpose(bls_data[:,pos,:]), marker='o') # -bls_data[f_RF_in,len(loc)-1,:] 
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
# plt.xlim(10,17.5)
# plt.ylim(-0.1,1e3)
# plt.ylim(-0.1,bls_data[f_RF_in,pos,f_bls_in]+10)
# plt.legend([str(f_RF[f_RF_in]) + ' GHz'])
plt.legend(leg)



#%% BLS vs. RF frequency

# plt.figure(num=200)
# plt.clf()
# ax = plt.imshow(bls_data[:,pos,:]-bls_data[:,0,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.max(f_RF),np.min(f_RF)])
# plt.xlabel('BLS Frequency (GHz)', fontsize=14)
# plt.ylabel('MW Frequency (GHz)', fontsize=14)

# plt.figure(num=201)
# plt.clf()
# ax = plt.imshow(bls_data[:,pos,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.max(f_RF),np.min(f_RF)])
# # plt.colorbar(ax)
# # plt.xlim(3,11)
# plt.xlabel('BLS Frequency (GHz)', fontsize=14)
# plt.ylabel('MW Frequency (GHz)', fontsize=14)
# plt.clim(0,2e1)


#%% position vs. Intensity

locBLS = []
Raleigh = []

for i in range(len(f_bls_in)):
    locBLS.append(bls_data[i,:,f_bls_in[i]])
    Raleigh.append(np.max(bls_data[f_RF_in[i],:,:], axis=1))
    
locBLS = np.transpose(np.array(locBLS))
Raleigh = np.transpose(np.array(Raleigh))

thermBLS = locBLS[0,:] # subtracting thermal signal at farthest position
thermBLS = thermBLS.astype('f')

netBLS = (locBLS-thermBLS)

netBLS = netBLS/Raleigh # normalize to Raleigh peak

plt.figure(num=300)
plt.clf()
ax = plt.plot(y_vec, netBLS, marker='o')
# ax = plt.plot(loc, bls_data[f_RF_in,:,f_bls_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend(leg)

#%% beam edge convolution

a_right = 0.05#np.min(daq_mean)
a_left = np.max(beam_ref)

edge_loc = 0.7
fwhm = np.array([0.7])
scan_edge = dgc.edge(y_vec,edge_loc, a_left, a_right)
[gauss,gauss_edge,Gint] = dgc.Gauss_conv(y_vec[:,0],fwhm,edge_loc, a_left, a_right)


plt.figure(num=301)
plt.clf()
ax = plt.plot(y_vec, beam_ref, marker='o')
plt.plot(y_vec,Gint[0,:]*a_left, marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('daq voltage', fontsize=14)
plt.legend(leg)

#%% closing the file
hFile = h5py.File(path)
hFile.close()

