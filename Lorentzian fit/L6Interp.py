# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:57:21 2023

@author: jjw7
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import csv
import h5py
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lmfit import models
from lmfit import Model
import imageio.v2 as imageio


def lorentz_sixth(x,x_0,gam,amp):
    return (amp/np.pi* gam / ( (x-x_0)**2 + gam**2 ))**6

def L6_peak_height(amp,gam):
    return (amp/gam/np.pi)**6

def L6_peak_err(amp,gam,amp_err,gam_err):
    return np.abs(6/np.pi**6*amp**5* (gam*amp_err-amp*gam_err)/gam**7)

def csv_write_to_fn(fn,arr):
    with open(fn, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(arr)

    csvFile.close()


plt.rcParams['axes.linewidth'] = 3
plt.rcParams.update({'figure.max_open_warning': 0})


#%% Function

"""
Function takes a normalized BLS spectra with a fitting window (and initial param guesses).
It interpolates a background, Fits to a Lorentzian^6, and outputs the fit parameters
"""


def BLS_L6fit_interp(blsf_meas,blsint_meas,sig_li,sig_ui,sig_cent,sig_width,interp_type = 'linear'):

    
    sig_bg_sub = np.zeros(blsint_meas.shape)
    bls_amp_interp_err_L6 = np.zeros((1,2))
    bls_gam_interp_err_L6 = np.zeros((1,2))
    bls_cent_interp_err_L6 = np.zeros((1,2))
    peak_height_L6 = np.zeros((1,2))
    
       
    # This bit just takes out the signal before interpolation    
    
    blsf_meas_bg = np.delete(blsf_meas,np.arange(sig_li,sig_ui,1))
    blsint_meas_bg = np.delete(blsint_meas,np.arange(sig_li,sig_ui,1))
 
    
    bg_interp = interp1d(blsf_meas_bg, blsint_meas_bg, kind=interp_type)
    
    # In case you want to look at the data with the signal removed
    
    # blsf_interp = np.linspace(blsf_meas[0],blsf_meas[-1],101, endpoint = True)
    # plt.figure(figsize = (14,10))
    # plt.scatter(blsf_meas,blsint_meas)
    # plt.scatter(blsf_meas_bg,blsint_meas_bg)
    # plt.plot(blsf_interp,bg_interp(blsf_interp))
  
    
    
    #sig_bg_sub = (blsint_meas - bg_interp(blsf_meas))/ bg_interp(blsf_meas)
  
    
    sig_bg_sub = (blsint_meas - bg_interp(blsf_meas))
    
    
    blsf_meas = blsf_meas[~np.isnan(sig_bg_sub).any(axis=0), :]
    blsint_meas = blsint_meas[~np.isnan(sig_bg_sub).any(axis=0), :]
    sig_bg_sub = sig_bg_sub[~np.isnan(sig_bg_sub).any(axis=0), :]
    # Now fitting the peak without the background    
    

    
    model_interp_L6 = Model(lorentz_sixth,prefix='L6_')
    model_interp_L6.set_param_hint('L6_x_0', min = sig_cent-sig_width/2, max = sig_cent+sig_width/2)
    params_interp_L6 = model_interp_L6.make_params(x_0=sig_cent, gam=1,amp = 10)
    output_interp_L6 = model_interp_L6.fit(sig_bg_sub,params_interp_L6, x=blsf_meas)
    components_interp_L6 = output_interp_L6.eval_components(x= blsf_meas) 
   
    
    # if you want to look at the fit without the background
    
    # fig1, ax = plt.subplots()
    # #plt.title(fn)
    # ax.scatter(blsf_meas,sig_bg_sub)
    # ax.plot(blsf_meas, components_interp_L6['mint_'])
  
    for param in output_interp_L6.params.values():
        if param.name == 'L6_amp':
            bls_amp_interp_err_L6[0,0] = param.value
            bls_amp_interp_err_L6[0,1] = param.stderr
            
        if param.name == 'L6_gam':
            bls_gam_interp_err_L6[0,0] = param.value
            bls_gam_interp_err_L6[0,1] = param.stderr
    
        if param.name == 'L6_x_0':
            bls_cent_interp_err_L6[0,0] = param.value
            bls_cent_interp_err_L6[0,1] = param.stderr    
    
    
    # plt.figure(figsize=(14,10))
    # plt.scatter(blsf_meas,blsint_meas)
    # plt.plot(blsf_meas, components_interp_L6['L6_']*bg_interp(blsf_meas)+ bg_interp(blsf_meas))
    # plt.ylabel('Intensity', fontsize=24)
    # plt.xlabel('Frequency (GHz)', fontsize=24)
    # plt.yscale('log')
    # #plt.title(fn[0:-3]+'_Position'+str(n), fontsize=24)
    # plt.tick_params(axis='x',length=16,width=3,labelsize=18)
    # plt.tick_params(axis='y',which='both',length=16,width=3,labelsize=18)
    # plt.tight_layout()
            
    
    
    peak_height_L6[:,0] = L6_peak_height(bls_amp_interp_err_L6[:,0], bls_gam_interp_err_L6[:,0])
    peak_height_L6[:,1] = L6_peak_err(bls_amp_interp_err_L6[:,0], bls_gam_interp_err_L6[:,0],bls_amp_interp_err_L6[:,1], bls_gam_interp_err_L6[:,1])
    
    return [bls_amp_interp_err_L6, bls_gam_interp_err_L6, bls_cent_interp_err_L6, peak_height_L6]                   




def BLS_L6fit(blsf_meas,blsint_meas,sig_li,sig_ui,sig_cent,sig_width,interp_type = 'linear'):

    
    bls_amp_interp_err_L6 = np.zeros((1,2))
    bls_gam_interp_err_L6 = np.zeros((1,2))
    bls_cent_interp_err_L6 = np.zeros((1,2))
    peak_height_L6 = np.zeros((1,2))
    
       
    

    
    model_interp_L6 = Model(lorentz_sixth,prefix='L6_')
    model_interp_L6.set_param_hint('L6_x_0', min = sig_cent-sig_width/2, max = sig_cent+sig_width/2)
    params_interp_L6 = model_interp_L6.make_params(x_0=sig_cent, gam=1,amp = 10)
    output_interp_L6 = model_interp_L6.fit(blsint_meas,params_interp_L6, x=blsf_meas)
    components_interp_L6 = output_interp_L6.eval_components(x= blsf_meas) 
   
    
    
    for param in output_interp_L6.params.values():
        if param.name == 'L6_amp':
            bls_amp_interp_err_L6[0,0] = param.value
            bls_amp_interp_err_L6[0,1] = param.stderr
            
        if param.name == 'L6_gam':
            bls_gam_interp_err_L6[0,0] = param.value
            bls_gam_interp_err_L6[0,1] = param.stderr
    
        if param.name == 'L6_x_0':
            bls_cent_interp_err_L6[0,0] = param.value
            bls_cent_interp_err_L6[0,1] = param.stderr    
    
    
    
    peak_height_L6[:,0] = L6_peak_height(bls_amp_interp_err_L6[:,0], bls_gam_interp_err_L6[:,0])
    peak_height_L6[:,1] = L6_peak_err(bls_amp_interp_err_L6[:,0], bls_gam_interp_err_L6[:,0],bls_amp_interp_err_L6[:,1], bls_gam_interp_err_L6[:,1])
    
    return [bls_amp_interp_err_L6, bls_gam_interp_err_L6, bls_cent_interp_err_L6, peak_height_L6]                   





















