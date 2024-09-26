# reading time-tagger bls files with Thatec package

import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
from matplotlib import pyplot as plt

#%% 


# path = r'O:\public\Martina Kiechle\BLS\Justins_amp.h5'
# path = r'O:\public\Martina Kiechle\BLS\Amp1_800nm\fsweep.h5'
# path = r'O:\public\Martina Kiechle\BLS\Amp2_600nm\Amp2_fsweep_600nm.h5'
# path = r'O:\public\Martina Kiechle\BLS\Amp2_600nm\Amp2_fsweep_600nm_3A.h5'
# path = r'O:\public\Martina Kiechle\BLS\Amp2_600nm\Amp2_fsweep_600nm_n4A.h5'
path = r'O:\public\Martina Kiechle\BLS\Amp2_600nm\Amp2_fsweep_600nm_n4A_fine.h5'



mt = MeasurementTree(path)
d = mt.dataset
print(d)


# f_RF = d['Frequency_1']
# frq = d['Frequency']
# dim = d['ScanDimension_1']
da = d['Acquire spectrum']
#ds = d['Get Data']


#%%

numfrq = d.Frequency.segments 
numfrf = d.Frequency_1.segments
# numpoin = d.ScanDimension_1.segments 
# numreps = d.repetitions.segments
# numtim = d.time.segments
# time = d.time.values
frq = d.Frequency.values[:,None]  
f_RF = d.Frequency_1.values[:,None] 
# loc = d.ScanDimension_1.values 
bls_data = da.values
bls_data_norm = da.values/np.max(da.values, axis=1)[:,None]

leg = [str(item)+' GHz' for item in f_RF[:,0]]


#%% offset between RF and BLS frequ.

frq_offset = np.zeros(len(f_RF))[:,None] 

blsF = f_RF-frq_offset #bls spectrum offset = -0.25 GHz atm

f_RF_in = [] 
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i,:]-f_RF[a,:]))) # 
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i,:]-blsF[a,:])))
    

#%% plot one slice

f_RF_in = 3

plt.figure(num=1)
plt.plot(frq,np.transpose(bls_data[:,:]), marker='o')
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.ylim(0,100)
plt.legend([str(f_RF[f_RF_in]) + ' GHz'])
plt.legend(leg)

#%%

plt.figure(num=2)
ax = plt.imshow(bls_data[:,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.min(f_RF),np.max(f_RF)])
plt.colorbar(ax)
plt.xlabel('BLS Frequency (GHz)', fontsize=14)
plt.ylabel('MW Frequency (GHz)', fontsize=14)
plt.clim(-0.001,50)


#%% closing the file
hFile = h5py.File(path)
hFile.close()

#%%
       
       
## pcolor uses too much rendering

# plt.figure(figsize=(8,5))
# ax = plt.pcolor(frq,np.flip(time),bls_avg[pos,:,:],shading='nearest')
# plt.colorbar(ax)
# plt.xlabel('Frequency (GHz)', fontsize=14)
# plt.ylabel('Time ($\mu$s)', fontsize=14)
# plt.clim(-0.001,10)
# plt.gca().invert_yaxis()

## caxis/clim in contourf does not work

# plt.figure(figsize=(8,5))
# ax = plt.contourf(frq,time,bls_avg[pos,:,:],vmin=0.01,vmax=10, extent = [np.min(frq),np.max(frq),np.max(time),np.min(time)]) # 
# ax = plt.contourf(frq,time,bls_avg[pos,:,:], extent = [np.min(frq),np.max(frq),np.max(time),np.min(time)]) # 
# plt.colorbar()
# plt.clim(-0.001,10)
# plt.xlabel('Frequency (GHz)', fontsize=14)
# plt.ylabel('Time ($\mu$s)', fontsize=14)
#plt.xlim(-14,-5)