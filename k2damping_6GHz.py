# read frequency sweep vs. amplitude

import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as ptc

#%% 

path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\150nm\cell9_150nm_fsweep_2A_m25dBm_10mW_6GHz.h5'


mt = MeasurementTree(path)
d = mt.dataset
print(d)


da = d['Acquire spectrum']
#ds = d['Get Data']


#%%

numfrq = d.Frequency.segments 
numfrf = d.Frequency_1.segments
# numpoin = d.ScanDimension_1.segments 
#numreps = d.repetitions.segments
#numtim = d.time.segments
#time = d.time.values
frq = d.Frequency.values
f_RF = d.Frequency_1.values
loc = d.ScanDimension_1_1.values 
bls_data = da.values
bls_data = bls_data.astype('f')

numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

# dx = d['ROI - relative positioning'].values[0,0]
# dy = d['ROI - relative positioning'].values[0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
# y_step = (y_stop-y_start)/(numpoin-1)
y_step = (y_stop-y_start)/(100-1)

# y_vec = np.arange(0, (y_stop-y_start)+y_step, y_step)
y_vec = loc*y_step
y_vec = np.round(y_vec,12) # 0 is not 0


#%% choosing data

pos = 25
extF = 6
frq_offset = 0.25

blsF = extF-frq_offset #bls spectrum offset = -0.25 GHz atm

# np.where(f_RF==extF)

f_RF_in = min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-extF))
f_bls_in = min(range(len(frq)), key=lambda i: abs(frq[i]-blsF)) # finding the index closest to f_RF in BLS frequencies



#%% plot one slice

plt.figure(num=200)
plt.plot(frq,bls_data[f_RF_in,pos,:], marker='o') # -bls_data[f_RF_in,len(loc)-1,:]
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
# plt.ylim(-0.1,5e4)
plt.ylim(-0.1,bls_data[f_RF_in,pos,f_bls_in]+10)
plt.legend([str(f_RF[f_RF_in]) + ' GHz'])


#%% position vs. Intensity

locBLS = bls_data[f_RF_in,:,f_bls_in] # BLS data vs. location for a certain frequ.

thermBLS = bls_data[f_RF_in,25,f_bls_in] # subtracting thermal signal at farthest position
# thermBLS = bls_data[abs(f_RF_in-1),:,f_bls_in] # subtracting thermal signal at different frequency

thermBLS = thermBLS.astype('f')
netBLS = locBLS-thermBLS

plt.figure(num=306)
ax = plt.plot(y_vec, netBLS, marker='o')
# ax = plt.plot(loc, bls_data[f_RF_in,:,f_bls_in], marker='o')
plt.xlabel('Location index', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend([str(f_RF[f_RF_in]) + ' GHz'])

fig, ax = plt.subplots()
strip = ptc.Rectangle((2.5,-50), 0.15, 700, facecolor='grey')
# plt.figure(num=301)
plt.plot(y_vec, netBLS, marker='o')
ax.add_patch(strip)
# ax = plt.plot(loc, bls_data[f_RF_in,:,f_bls_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend([str(f_RF[f_RF_in]) + ' GHz'])
plt.xlim(0,5)



#%% closing the file
hFile = h5py.File(path)
hFile.close()

