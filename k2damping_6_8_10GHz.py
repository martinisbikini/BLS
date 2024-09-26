# read frequency sweep vs. amplitude


import xarray as xr
import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
# %matplotlib qt
from matplotlib import pyplot as plt
from matplotlib import patches as ptc

#%% 

"10 GHZ was not recorded due to a measurement interruption"

path = r'O:\public\Martina Kiechle\BLS\k2_damping\Cell9\150nm\cell9_150nm_fsweep_2A_m20dBm_10mW_6_8_10.h5'


mt = MeasurementTree(path)
d = mt.dataset
print(d)

da = d['Acquire spectrum']


#%%

numfrq = d.Frequency.segments 
numfrf = d.Frequency_1.segments
# numpoin = d.ScanDimension_1.segments 
#numreps = d.repetitions.segments
#numtim = d.time.segments
#time = d.time.values
frq = d.Frequency.values[:,None] 
f_RF = d.Frequency_1.values[:,None] 
loc = d.ScanDimension_1_1.values[:,None] 
bls_data = da.values.astype('f')


bls_data

numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

# dx = d['ROI - relative positioning'].values[0,0]
# dy = d['ROI - relative positioning'].values[0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(100-1)


y_vec = loc*y_step
y_vec = np.round(y_vec,12) # 0 is not 0

leg = [str(item)+' GHz' for item in f_RF[:,0]]


#%% choosing data

pos = 40

frq_offset = np.array([-0.125,-0.125,-0.25])[:,None]

blsF = f_RF+frq_offset #bls spectrum offset = -0.25 GHz atm

f_RF_in = [] #np.linspace(0, len(f_RF)-1,len(f_RF))
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
    
    locBLS.append(np.sum(bls_data[i,:,f_bls_in[i]-2:f_bls_in[i]+2],axis=1))
    
    # locBLS.append(bls_data[i,:,f_bls_in[i]])
    Raleigh.append(np.max(bls_data[f_RF_in[i],:,:], axis=1))
    
    
locBLS = np.transpose(np.array(locBLS))
Raleigh = np.transpose(np.array(Raleigh))

thermBLS = locBLS[0,:].astype('f') # subtracting thermal signal at farthest position
# netBLS = (locBLS-thermBLS)/Raleigh # normalize to Raleigh peak
netBLS = (locBLS)/Raleigh # normalize to Raleigh peak


#%% position plots

# all frequ.
plt.figure(num=300)
plt.clf()
ax = plt.plot(y_vec, netBLS, marker='o')
# ax = plt.plot(loc, bls_data[f_RF_in,:,f_bls_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend(leg)


# selected frequ.
fplot = 6
f_plot_in = min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-fplot))

fig, ax = plt.subplots(num=301)
plt.clf()
strip = ptc.Rectangle((1.15,-50), 0.15, 700, facecolor='grey')
# plt.figure(num=301)
plt.plot(y_vec, netBLS[:,f_plot_in], marker='o')
ax.add_patch(strip)
# ax = plt.plot(loc, bls_data[f_RF_in,:,f_bls_in], marker='o')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend([leg[f_plot_in]])


#%% closing the file
hFile = h5py.File(path)
hFile.close()

