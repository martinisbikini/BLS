
import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('O:\public\Martina Kiechle\BLS\Data analysis')
import edge_gauss_convolution as dgc

#%% 


path = r'O:\public\Martina Kiechle\edgescan\edgescan\yscan_10_10x.h5'


mt = MeasurementTree(path)
d = mt.dataset

loc = d.ScanDimension_1.values
daq = d['Ch 4'].values # reps, steps, samples


daq1 = np.squeeze(daq[0,:,:])

for i in range(0,9):
    daq1 = np.append(daq1,np.squeeze(daq[i,:,:]), axis=1)
    
daq = daq1
   
daq_mean = np.mean(daq, axis=1)


#%% line length

numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(numpoin-1)

y_vec = np.arange(0, (y_stop-y_start)+y_step, y_step)
y_vec = np.round(y_vec,12) # 0 is not 0


#%% edge

x1 = np.arange(-1, (y_stop-y_start)+y_step+1, y_step)
x1 = np.round(x1,12) # 0 is not 0

e_min = 0.33#np.min(daq_mean)
e_max = np.max(daq_mean)

edge_loc = 1.7
fwhm = np.array([0.7])

scan_edge = dgc.edge(x1,edge_loc, e_min, e_max)
[gauss,gauss_edge,Gint] = dgc.Gauss_conv(x1,fwhm,edge_loc, e_min, e_max)

# plt.figure(num=3)
# plt.plot(x1,scan_edge, marker='o')
# plt.plot(x1,Gint[0,:], marker='o')

#%% data

plt.figure(num=1)
plt.plot(y_vec,daq[:,:], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Daq voltage', fontsize=14)


plt.figure(num=2)
plt.plot(y_vec,daq_mean, marker='o')
plt.plot(x1,Gint[0,:]*e_max, marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Daq voltage', fontsize=14)
plt.xlim(0, y_stop+y_step)















