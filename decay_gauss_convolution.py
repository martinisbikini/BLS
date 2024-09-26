""" Convolution of Gaussian beam shape with exponential decay """

import numpy as np
import matplotlib.pyplot as plt

wstrip = 2
s = 0.05
x1 = np.arange(-5, 5+s, s)
x1 = np.round(x1,12) # 0 is not 0
# x1 = x1[:,None] # 0 is not 0
A = 1
B = A*1
fwhm = np.array([0.4, 0.8, 1.2, 1.6])
alpha = 1.5
ld = 0.54
l_edge = -2 
r_edge = 0

#%%  left edge

# def SWdecay(x,wstrip,A,B,l_edge,ld):
#     # return np.piecewise(x, [x <= -wstrip, (x > -wstrip) & (x < 0), x >= 0], [lambda x: B*np.exp(-(abs(x)-wstrip)/ld), 0, lambda x: A*np.exp(-(abs(x))/ld)])
#     # return np.piecewise(x, [x <= -wstrip/2, (-wstrip/2 < x)  & (x < wstrip/2), x >= wstrip/2], [lambda x: B*np.exp(-(abs(x)-wstrip/2)/ld), 0, lambda x: A*np.exp(-(abs(x)-wstrip/2)/ld)]) 
#     # return np.piecewise(x, [x <= l_edge, (l_edge < x)  & (x < wstrip+l_edge), x >= wstrip+l_edge], [lambda x: B*np.exp((-abs(x-l_edge))/ld), 0, lambda x: A*np.exp((-abs(x-l_edge)+wstrip)/ld)])
#     return np.piecewise(x, [x <= l_edge, (l_edge < x)  & (x < wstrip+l_edge), x >= wstrip+l_edge], [lambda x: B*np.exp((-abs(x-l_edge))/ld), 0, lambda x: A*np.exp((-abs(x-l_edge)+wstrip)/ld)])

#%%

def SWdecay(x,wstrip,A,B,r_edge,ld):
    return np.piecewise(x, [x <= r_edge-wstrip, (r_edge-wstrip < x)  & (x < r_edge), x >= r_edge], [lambda x: B*np.exp(((-abs(x-r_edge)+wstrip))/ld), 0, lambda x: A*np.exp(-abs(x-r_edge)/ld)])


def Gauss_conv(x,fwhm,wstrip,A,B,r_edge,ld):
    gauss = np.zeros((len(fwhm), len(x), len(x)))
    gauss_decay = np.zeros((len(fwhm), len(x), len(x)))
    leg = []
    sig = fwhm/2.355
    
    for j in range(len(fwhm)):  
        leg.append('FWHM = ' + str(fwhm[j])) 
        
        for i in range(len(x)):
            gauss[j, i, :] = np.exp(-(x-x[i])**2/(2*sig[j]**2)) 
            gauss_decay[j, i, :] = SWdecay(x,wstrip,A,B,r_edge,ld) * gauss[j, i, :]
            
    Gint = np.trapz(gauss_decay, axis=2)
    Gmax = np.max(Gint, axis=1)
    Gind = np.argmax(Gint, axis=1)
    Gint = np.divide(Gint,Gmax[:,None])

    return gauss,gauss_decay,Gint




def Gauss_exp(x,fwhm,wstrip,A,B,r_edge,ld,y_off):
    
    sig = fwhm/2.355
    gauss = np.zeros((len(x), len(x)))
    gauss_decay = np.zeros((len(x), len(x)))
    
    for i in range(len(x)):
        
        gauss[i,:] = np.exp(-(x-x[i])**2/(2*sig**2)) 
        gauss_decay[i,:] = SWdecay(x,wstrip,A,B,r_edge,ld) * gauss[i, :] + y_off
        
    Gint = np.trapz(gauss_decay, axis=1)
    Gmax = np.max(Gint, axis=0)
    # Gint = np.divide(Gint,Gmax)
        
    return Gint 


#%%

decay = SWdecay(x1, wstrip, A, B, r_edge, ld)
[gauss,gauss_decay,Gint] = Gauss_conv(x1,fwhm,wstrip,A,B,r_edge,ld)
Gexp = Gauss_exp(x1, fwhm[0], wstrip, A, B, r_edge, ld,0)

fwhm_in = 1


fig = plt.figure(num=6, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)
ax.plot(x1,gauss[0,:,:]*decay)
ax.plot(x1,decay, linestyle='dashed', linewidth=2, color='black', label='Exp decay')
# ax.plot(x1,Gint[fwhm_in,:], linestyle=None, color='black',linewidth=3)
    
left, bottom, width, height = 0.15, 0.15, 0.75, 0.75
ax.set_position([left, bottom, width, height])

# plt.legend(['Exp decay', 'Gauss-exp conv.'])
plt.title('ld = '+str(ld)+' um, FWHM = '+str(fwhm[fwhm_in]))
plt.xlim(-1, 3.3)
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Amplitude (normalized)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


fig = plt.figure(num=7, figsize=(7,5))
plt.clf()
ax = fig.add_subplot(111)
ax.plot(x1[:,None],np.transpose(Gint[:,:]), marker='o')
plt.legend([str(ele) for ele in fwhm], title='FWHM ($\mu$m)', fontsize=16)

left, bottom, width, height = 0.15, 0.15, 0.5, 0.5
ax.set_position([left, bottom, width, height])

# plt.title('Gauss-exp decay convolution')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.xlim(-1, 3.3)
plt.xlabel('Distance ($\mu$m)', fontsize=16)
plt.ylabel('Amplitude (normalized)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig('Gaus_exp_fwhm.svg', format='svg', transparent=True)

plt.figure(num=8)
plt.clf()
plt.plot(x1[:,None],Gexp[:,None], marker='o')
plt.legend([str(ele) for ele in fwhm])
plt.title('Gauss-exp decay convolution')
plt.xlabel('Distance ($\mu$m)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.xlim(-1.2,4.2)
