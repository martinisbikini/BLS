""" Convolution of Gaussian beam shape with exponential decay """

import numpy as np
import matplotlib.pyplot as plt

#%%

# edge_loc = -0.5
# s = 0.025
# x1 = np.arange(-5, 5+s, s)
# x1 = np.round(x1,12) # 0 is not 0
# fwhm = np.array([2])


#%%

def edge(x,edge_loc, left, right):
    return np.piecewise(x,[x < edge_loc, x >= edge_loc], [left, right])


def Gauss_conv(x,fwhm,edge_loc, left, right):
    gauss = np.zeros((len(fwhm), len(x), len(x)))
    gauss_edge = np.zeros((len(fwhm), len(x), len(x)))
    leg = []
    sig = fwhm/2.355
    
    for j in range(len(fwhm)):  
        leg.append('FWHM = ' + str(fwhm[j])) 
        
        for i in range(len(x)):
            gauss[j, i, :] = np.exp(-(x-x[i])**2/(2*sig[j]**2))
            gauss_edge[j, i, :] = edge(x,edge_loc, left, right) * gauss[j, i, :]
            
    Gint = np.trapz(gauss_edge, axis=2)
    Gmax = np.max(Gint, axis=1)
    Gind = np.argmax(Gint, axis=1)
    Gint = np.divide(Gint,Gmax[:,None])

    return gauss,gauss_edge,Gint



#%%

# scan_edge = edge(x1, edge_loc,1,-1)
# [gauss,gauss_edge,Gint] = Gauss_conv(x1,fwhm,edge_loc,1,-1)

# plt.figure(num=3)
# plt.clf()
# plt.plot(x1,scan_edge, marker='o')
# plt.plot(x1,Gint[0,:], marker='o')
# plt.legend(['Edge function', 'pt-wise Gauss-edge convolution'])
 


