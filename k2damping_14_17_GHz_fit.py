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
import scipy.optimize as opt
import matplotlib.patches as patches


#%% 

"10 GHZ was not recorded due to a measurement interruption"


# path = r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_8dBm_10mW_14to17_2p5um.h5'
# path = r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_10dBm_10mW_20to23_2p5um.h5'
path = r'Y:\Martina Kiechle\BLS\k2_damping\Cell9\2000nm\cell9_2um_fsweep_4A_15dBm_10mW_26to33_2p5um.h5'


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
fwhm = np.array([0.91])

fplot = f_RF[2,0] # GHz
f_plot_in = f_RF[:,0].tolist().index(fplot)

f_plot_in = min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-fplot)) # find index

f_wind = 2 # GHz

# frq_offset = np.array([-0.5,-0.5])[:,None] #bls spectrum offset to RF frequ.
frq_offset = np.array([-0.5,-0.5,-0.625])[:,None] #bls spectrum offset to RF frequ.

blsF = f_RF+frq_offset 

f_RF_in = [] 
f_bls_in = []

for a in range(len(f_RF)):
    f_RF_in.append(min(range(len(f_RF)), key=lambda i: abs(f_RF[i]-f_RF[a])))
    f_bls_in.append(min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])))
    # f_bls_in[a] = min(range(len(frq)), key=lambda i: abs(frq[i]-blsF[a])) # finding the index closest to f_RF in BLS frequencies


#%% plot intensity of locations

# x_min_ind = frq[:,0].tolist().index(fplot-f_wind/2)
# x_max_ind = frq[:,0].tolist().index(fplot+f_wind/2)

x_min = frq[f_bls_in[f_plot_in]]-f_wind/2
x_max = frq[f_bls_in[f_plot_in]]+f_wind/2

# x_min_ind = min(range(len(frq)), key=lambda i: abs(x_min))
# x_max_ind = min(range(len(frq)), key=lambda i: abs(x_max))

y_min = 0
y_max = np.max(bls_data[f_plot_in,:,f_bls_in[f_plot_in]-10:f_bls_in[f_plot_in]+10])


#%% position vs. Intensity

locBLS = []
Raleigh = []


for i in range(len(f_bls_in)):
    
    locBLS.append(np.sum(bls_data[i,:,f_bls_in[i]-1:f_bls_in[i]+1],axis=1)) # build sum of nearest peak frequ. 
    Raleigh.append(np.max(bls_data[f_RF_in[i],:,:], axis=1))
      
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
# ld = 0.6
A = 1
B = 1
l_edge = edge_loc
r_edge = edge_loc


#%% fit


def Gauss_exp(x,fwhm,ld,y_off):
    
    sig = fwhm/2.355
    gauss = np.zeros((len(x), len(x)))
    gauss_decay = np.zeros((len(x), len(x)))
    # ld = k**2 * ld
    
    SWdecay = np.piecewise(x, [x <= r_edge-wstrip, (r_edge-wstrip < x)  & (x < r_edge), x >= r_edge], [lambda x: B*np.exp(((-abs(x-r_edge)+wstrip))/ld), 0, lambda x: A*np.exp(-abs(x-r_edge)/ld)])
    
    for i in range(len(x)):
        
        gauss[i,:] = np.exp(-(x-x[i])**2/(2*sig**2)) 
        gauss_decay[i,:] = SWdecay * gauss[i, :] + y_off
        
    Gint = np.trapz(gauss_decay, axis=1)
    Gmax = np.max(Gint, axis=0)
    Gint = np.divide(Gint,Gmax)
        
    return Gint 


corr_sig = (netBLS[:,f_plot_in]-np.min(netBLS[:,f_plot_in]))/max((netBLS[:,f_plot_in]-np.min(netBLS[:,f_plot_in])))

popt, pcov = opt.curve_fit(Gauss_exp, np.squeeze(y_vec), corr_sig) # remove dimension

#%%

waveguide = patches.Rectangle((0-edge_loc, 0), edge_loc, 1.1, linewidth=1, edgecolor=None, facecolor='gold', alpha=0.2, label='waveguide')
cofe = patches.Rectangle((0, 0), 3.3, 1.1, linewidth=1, edgecolor=None, facecolor='gray', alpha=0.2, label='cofe')

fig, ax = plt.subplots(num=303, figsize=(7,5))
plt.clf()


ax = fig.add_subplot(111)

ax.add_patch(waveguide)
ax.add_patch(cofe)

SWsig, = ax.plot(y_vec-edge_loc, corr_sig, marker='o', label=leg[f_plot_in], color='blue')
wgedge = plt.vlines(0, np.min(netBLS[:,f_plot_in])*0.9*0, np.max(corr_sig)*1.1,color='black',linestyle='dashed',label='Waveguide edge')
GEfit, = ax.plot(np.squeeze(y_vec-edge_loc),Gauss_exp(np.squeeze(y_vec),*popt),linewidth=2, color='red', label='Gauss * exp')

left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])

plt.xlim(0-edge_loc,4-edge_loc)
plt.ylim(np.min(netBLS[:,f_plot_in])*0,np.max(corr_sig)*1.1)


plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Photon counts (normalized)', fontsize=16)


plt.legend(handles=[SWsig, GEfit],fontsize=16)
plt.title('FWHM = '+str(round(popt[0],2))+', ld = '+str(round(popt[1],2))+' um')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

save_path = 'Y:/Martina Kiechle/Conferences/Poster/'
plt.savefig(save_path+'GaussFit_'+str(fplot)+'GHz.svg', format='svg')


#%% closing the file
hFile = h5py.File(path)
hFile.close()

