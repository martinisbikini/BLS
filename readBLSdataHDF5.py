# reading time-tagger bls files with Thatec package

import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np

#%% 

path = r'O:\public\Martina Kiechle\BLS\Justins_amp.h5'

#%% 

hdf =  h5py.File(path, 'r')
keys = list(hdf.keys())

ttg_dat0 = list(hdf['measurement/row_11/data'])
ttg_scale = list(hdf['measurement/row_11/scale'])
ttg_info = list(hdf['devices/TimeTagger 2.10.6'])
scan_len = list(hdf['measurement/row_02/data'])
scan_len = float(scan_len[0])

tree = list(hdf['scan_definition/tree_view'])
reps = list(hdf['scan_definition/row_04'])

#%% 

numreps = reps[2]

numsteps = int(len(ttg_dat0)/numreps)


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