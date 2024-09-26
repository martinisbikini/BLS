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


path = r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_10dBm_10mW_20to23_2p5um.h5'


mt = MeasurementTree(path)
d = mt.dataset
print(d)

da = d['Acquire spectrum']


#%%

numfrq = d.Frequency.segments 
numfrf = d.Frequency_1.segments
frq = d.Frequency.values[:,None] 
f_RF = d.Frequency_1.values[:,None] 
loc = d.ScanDimension_1_1.values[:,None] 
bls_data = da.values.astype('f')

# positioning
numpoin = float(len(loc))
dx = d['ROI - relative positioning'].values[0,0,0]
dy = d['ROI - relative positioning'].values[0,0,1]

y_start = float(0)
y_stop = np.sqrt(dx**2+dy**2)
y_step = (y_stop-y_start)/(numpoin-1)

y_vec = (loc-np.min(loc))*y_step
y_vec = np.round(y_vec,12) # 0 is not 0


# frq legend
leg = [str(item)+' GHz' for item in f_RF[:,0]]


#%% choosing data

pos = 13

edge_loc = 0.8
fwhm = np.array([0.9])

fplot = f_RF[1,0] # GHz
f_plot_in = f_RF[:,0].tolist().index(fplot)

f_plot_in = min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-fplot)) # find index

f_wind = 2 # GHz

frq_offset = np.array([-0.5,-0.5])[:,None] #bls spectrum offset to RF frequ.

blsF = f_RF+frq_offset 

f_RF_in = [] 
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
# plt.xlim(10,17.5)
# plt.ylim(-0.1,5e4)
plt.ylim(-0.1,bls_data[f_RF_in[0],pos,f_bls_in[0]]+200)
plt.legend(leg)
plt.title('x = '+str(np.round(float(y_vec[pos]),3))+' '+chr(956)+'m')


#%% plot intensity of locations

# x_min_ind = frq[:,0].tolist().index(fplot-f_wind/2)
# x_max_ind = frq[:,0].tolist().index(fplot+f_wind/2)

x_min = frq[f_bls_in[f_plot_in]]-f_wind/2
x_max = frq[f_bls_in[f_plot_in]]+f_wind/2

# x_min_ind = min(range(len(frq)), key=lambda i: abs(x_min))
# x_max_ind = min(range(len(frq)), key=lambda i: abs(x_max))

y_min = 0
y_max = np.max(bls_data[f_plot_in,:,f_bls_in[f_plot_in]-10:f_bls_in[f_plot_in]+10])


plt.figure(num=101)
plt.clf()
plt.plot(frq,np.transpose(bls_data[f_plot_in,:,:]), marker='o')
plt.xlabel('Frequency (GHz)', fontsize=14)
plt.ylabel('BLS cts', fontsize=14)
plt.xlim(x_min,x_max)
plt.ylim(0,y_max*1.1)
# plt.ylim(-0.1,bls_data[f_RF_in[0],pos,f_bls_in[0]]+200)
# plt.legend(leg)
# plt.title('x = '+str(np.round(float(y_vec[pos]),3))+' '+chr(956)+'m')



#%% BLS vs. RF frequency


plt.figure(num=201)
plt.clf()
ax = plt.imshow(bls_data[:,pos,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.max(f_RF),np.min(f_RF)])
# ax = plt.imshow(bls_data[:,pos,:]-bls_data[:,0,:],aspect ='auto',extent = [np.min(frq),np.max(frq),np.max(f_RF),np.min(f_RF)])
plt.colorbar(ax)
plt.xlabel('BLS Frequency (GHz)', fontsize=14)
plt.ylabel('MW Frequency (GHz)', fontsize=14)
plt.clim(0,2e1)


#%% position vs. Intensity

locBLS = []
Raleigh = []


for i in range(len(f_bls_in)):
    
    locBLS.append(np.sum(bls_data[i,:,f_bls_in[i]-1:f_bls_in[i]+1],axis=1)) # build sum of nearest peak frequ. 
    Raleigh.append(np.max(bls_data[f_RF_in[i],:,:], axis=1))
    
    
# for i in range(len(f_bls_in)):   
    # locBLS.append(bls_data[i,:,f_bls_in[i]])
    # Raleigh.append(np.max(bls_data[f_RF_in[i],:,:], axis=1))
    
    
locBLS = np.transpose(np.array(locBLS))
Raleigh = np.transpose(np.array(Raleigh))

thermBLS = locBLS[0,:].astype('f') # subtracting thermal signal at farthest position
# netBLS = (locBLS-thermBLS)/Raleigh # normalize to Raleigh peak
netBLS = (locBLS)/Raleigh # normalize to Raleigh peak


#%% plot & fit daq voltage from photo diode

daq = d['Ch 4'].values # reps, steps, samples

plt.figure(num=51)
plt.clf()
plt.plot(y_vec,np.transpose(daq)-np.min(daq,axis=1), marker='o') 
# plt.plot(y_vec,np.transpose(daq-np.min(daq)), marker='o') 
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('DAQ (V)', fontsize=14)

# x1 = np.arange(-1, (y_stop-y_start)+y_step+1, y_step)
x1 = np.arange(-1, y_stop, y_step)
x1 = np.round(x1,12) # 0 is not 0

left = np.max(daq)-np.min(daq)
right =0


scan_edge = egc.edge(x1,edge_loc, left, right)
[gauss,gauss_edge,G_e_int] = egc.Gauss_conv(x1,fwhm,edge_loc, left, right)

plt.plot(x1,G_e_int[0,:]*left,linewidth=2,color='black') 
plt.plot(x1,scan_edge,linewidth=2,color='black',linestyle='dashed')

plt.legend(leg+['Gauss-edge convolution','Waveguide edge'])
plt.title('FWHM = '+str(float(fwhm))+' '+chr(956)+'m')


#%% Gauss convolution with exp. decay

wstrip = 2
ld = 0.6
A = 1
B = 1
l_egde = edge_loc

decay = dgc.SWdecay(x1, wstrip, A, B, l_egde, ld)
[gauss,gauss_decay,Gint] = dgc.Gauss_conv(x1,fwhm,wstrip,A,B,l_egde,ld)


#%% position plots

# all frequ.
plt.subplots(num=300)

plt.clf()
# plt.plot(y_vec, netBLS, marker='o')
plt.plot(y_vec-0.8, netBLS/max(netBLS[:,0]), marker='o')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
# plt.ylim(np.min(netBLS)*0.9,np.max(netBLS)*1.1)
plt.vlines(edge_loc, np.min(netBLS)*0.9, np.max(netBLS)*1.1,color='black',linestyle='dashed')
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Photon counts (normalized)', fontsize=16)
# plt.legend(leg+['Waveguide edge'])
plt.legend(leg, fontsize=16)
# plt.xlim(0,np.max(y_vec-0.8))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


# selected frequ.

fig, ax = plt.subplots(num=301)
plt.clf()
plt.plot(y_vec, netBLS[:,f_plot_in]-np.min(netBLS[:,f_plot_in]), marker='o')
plt.vlines(edge_loc, np.min(netBLS[:,f_plot_in])*0.9*0, np.max(netBLS[:,f_plot_in])*1.1,color='black',linestyle='dashed')
plt.ylim(np.min(netBLS[:,f_plot_in])*0,np.max(netBLS[:,f_plot_in])*1.1)

plt.plot(x1,Gint[0,:]*(np.max(netBLS[:,f_plot_in])-np.min(netBLS[:,f_plot_in])),linewidth=2,color='red')
plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts', fontsize=14)
plt.legend([leg[f_plot_in],'Waveguide edge','fitted Gauss * exp'])
plt.title(chr(955)+' = '+str(float(ld))+' '+chr(956)+'m, FWHM = '+str(float(fwhm))+' '+chr(956)+'m')


#%%

fig, ax = plt.subplots(num=302)
plt.clf()
bla = netBLS[:,f_plot_in]-np.min(netBLS[:,f_plot_in])

plt.plot(y_vec, bla/max(bla), marker='o')
plt.xlim(edge_loc,np.max(y_vec))

# plt.plot(x1,Gint[0,:]*(np.max(netBLS[:,f_plot_in])-np.min(netBLS[:,f_plot_in])),linewidth=2,color='red')
plt.plot(x1,Gint[0,:],linewidth=2,color='red')

plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Photon cts (normalized)', fontsize=14)
plt.legend([leg[f_plot_in],chr(955)+' = '+str(float(ld))+' '+chr(956)+'m'])


#%% closing the file
hFile = h5py.File(path)
hFile.close()

