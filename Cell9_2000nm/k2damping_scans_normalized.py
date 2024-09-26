# read frequency sweep vs. amplitude


import xarray as xr
import PyThat
from PyThat import MeasurementTree
import h5py
import numpy as np
# %matplotlib qt
from matplotlib import pyplot as plt
import sys
sys.path.append('Y:\Martina Kiechle\BLS\Data analysis')
import edge_gauss_convolution as egc
import decay_gauss_convolution as dgc

#%% 

"10 GHZ was not recorded due to a measurement interruption"

file = 2


path = [r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_8dBm_10mW_14to17_2p5um.h5',
         r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_10dBm_10mW_20to23_2p5um.h5',
         r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_15dBm_10mW_26to33_2p5um.h5']


Prf = [8,8,10,10,15,15,15]
Pleg = [str(item)+' dBm' for item in Prf]

#%% 
    
d = []
f_RF = []
f_RF1 = []
bls_data = []
frq = []
loc = []
dx = np.zeros(len(path))
dy = np.zeros(len(path))
numpoin = np.zeros(len(path))
numfrq = np.zeros(len(path)) 
# numfrf = np.zeros(len(path))
numfrf = []


for i in range(len(path)):
   
    mt = MeasurementTree(path[i])
    d.append(mt.dataset)
    f_RF = np.concatenate([f_RF, d[i].Frequency_1.values], axis=0) if 'f_RF' in locals() else f_RF1   
    f_RF1.append(d[i].Frequency_1.values)
    frq.append(d[i].Frequency.values[:,None])
    bls_data.append(d[i]['Acquire spectrum'].values.astype('f'))  
    dx[i] = d[i]['ROI - relative positioning'].values[0,0,0]
    dy[i] = d[i]['ROI - relative positioning'].values[0,0,1]
    loc.append(d[i].ScanDimension_1_1.values[:,None])
    numpoin[i] = float(len(loc[i]))
    # numfrq[i] = d[i].Frequency.segments 
    numfrf.append(d[i].Frequency_1.segments)
    

    
y_start = np.zeros(len(path))
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(numpoin-1)
y_vec = (loc-np.min(loc))*y_step
y_vec = np.round(y_vec,12) # 0 is not 0


# frq legend
leg = [str(item)+' GHz' for item in f_RF]

# offset to fRF in spectrum in GHz
frq_offset = np.array([-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.625])[:,None] 


#%% choosing data

f_RF = f_RF[:,None]

pos = 14
fplot = f_RF[2,0] # GHz
f_plot_in = f_RF[:,0].tolist().index(fplot)

f_plot_in = min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-fplot)) # find index

f_wind = 2 # GHz


blsF = f_RF+frq_offset 

f_RF_in = [] 
f_bls_in = []
iloop = []

locBLS = []
Raleigh = []
test = []
test1 = []


for a in range(len(numfrf)): 
    for b in range(numfrf[a]):
        iloop.append([a,b])
        c = len(iloop)-1
        f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[c])))
        f_bls_in.append(min(range(len(frq[a])), key=lambda i: abs(frq[a][i]-blsF[c])))
        
        locBLS.append(np.sum(bls_data[a][b,:,f_bls_in[c]-2:f_bls_in[c]+2],axis=1)) # build sum of nearest peak frequ.      
        # Raleigh.append(np.max(bls_data[a][f_RF_in[b],:,:], axis=1))
        Raleigh.append(np.max(bls_data[a][b,:,:], axis=1))        
        test.append(f_bls_in[b])
        test1.append(Raleigh)
        
        
        
#%% position vs. Intensity

# locBLS = []
# Raleigh = []
# test = []


# for j in range(len(numfrf)):
    
#     for i in range(numfrf[j]):    
        
#         locBLS.append(np.sum(bls_data[j][i,:,f_bls_in[i]-2:f_bls_in[i]+2],axis=1)) # build sum of nearest peak frequ.      
#         Raleigh.append(np.max(bls_data[j][f_RF_in[i],:,:], axis=1))        
#         test.append(f_bls_in[i])
    

locBLS = np.transpose(np.array(locBLS))
Raleigh = np.transpose(np.array(Raleigh))

thermBLS = locBLS[0,:].astype('f') # subtracting thermal signal at farthest position
# netBLS = (locBLS-thermBLS)/Raleigh # normalize to Raleigh peak
netBLS = (locBLS)/Raleigh # normalize to Raleigh peak        
        
        
    
#%% plot one slice

fig = plt.figure(num=990, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)

for i in range(len(numfrf)):
    ax.plot(frq[i],np.transpose(bls_data[i][:,pos,:]), marker='o')
    
ax.plot(frq[2]+21, bls_data[0][f_RF_in[0],pos,f_bls_in[0]] * np.exp((-frq[2]-7.5)*0.25), linestyle='dashed', linewidth=2)
    
left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])

plt.xlabel('Frequency (GHz)', fontsize=16)
plt.ylabel('BLS photon counts', fontsize=16)
plt.xlim(np.min(f_RF)-2.5,np.max(f_RF)+1)
# plt.ylim(-0.1,5e4)
plt.ylim(-0.1,bls_data[0][f_RF_in[0],pos,f_bls_in[0]]+50)

myleg = result = [f"{x}, {y}" for x, y in zip(leg, Pleg)]
plt.legend(myleg+['exp(-f/4)'], fontsize=16)


# plt.title('x = '+str(np.round(float(y_vec[pos]),3))+' '+chr(956)+'m')
plt.xticks(np.squeeze(f_RF,axis=1),fontsize=16)
plt.yticks(fontsize=16)



#%% position plots

# all frequ.
# plt.subplots(num=300)


fig = plt.figure(num=991, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)

for i in range(len(f_RF)):
    plt.plot(y_vec[0,:,0]-0.8, netBLS[:,i]/max(netBLS[:,0]), marker='o')

left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.ylim(np.min(netBLS)*0.9,np.max(netBLS)*1.1)
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Photon counts (normalized)', fontsize=16)
# plt.legend(leg+['Waveguide edge'])
plt.legend(leg, fontsize=16)
# plt.xlim(0,np.max(y_vec-0.8))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)



fig = plt.figure(num=992, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)

for i in range(0,7,1):
    plt.plot(y_vec[0,:,0]-0.8, netBLS[:,i], marker='o')

left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])    
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.ylim(np.min(netBLS)*0.9,np.max(netBLS)*1.1)
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Photon counts (normalized)', fontsize=16)
# plt.legend(leg+['Waveguide edge'])
plt.legend(leg, fontsize=16)
# plt.xlim(0,np.max(y_vec-0.8))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#%% closing the file
hFile = h5py.File(path[file])
hFile.close()

