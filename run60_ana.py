import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
from scipy.stats import binned_statistic

sys.path.append('/reg/neh/home4/espov/python/datastorage/')
import datastorage

import ana_fun as ana

""" ---------------------------------------------------------------- """


data = datastorage.read('vesp_run60_anaMPI.h5')

mask = np.logical_and(~np.isnan(data.sig), ~np.isnan(data.I0))

I0 = data.I0[mask]
I = data.sig[mask]
energy = data.ebeam.photon_energy[mask]
energyL3 = data.ebeam.L3_energy[mask]


""" I-I0 correlation """
plt.figure('I0 correlation')
plt.plot(I0, I, 'o', markersize=0.02)
plt.xlabel('I0')
plt.ylabel('I')

bins = np.arange(0,0.0014,0.0001)
med = binned_statistic(I0, I, bins=bins, statistic='median').statistic
xmed = bins[:-1]+np.diff(bins)/2
plt.plot(xmed, med, linewidth=2)
plt.show()



""" Energy spectrum analysis """
en_bins = np.arange(900,935,0.2)
xen = en_bins[:-1]+np.diff(en_bins)/2
IvsFEE_average = binned_statistic(energy,I, bins=en_bins, statistic='mean').statistic
I0vsFEE_average = binned_statistic(energy,I0, bins=en_bins, statistic='mean').statistic

plt.figure('Energy')
plt.plot(xen, IvsFEE_average/np.nanmax(IvsFEE_average), label='signal')
plt.plot(xen, I0vsFEE_average/np.nanmax(I0vsFEE_average), label='I0')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity')
plt.legend()
plt.show()



""" Linearization """
#energy_I0_product = (energy-np.mean(energy))*(I0-np.mean(I0))
#X = np.array([np.ones(len(I0)), I0, energy, energy_I0_product]).transpose()
#X_inv = np.matmul( np.linalg.inv(np.matmul(X.transpose(), X)) , X.transpose() ) 
# pseudo inverse (Moore-Penrose inverse)
#beta = np.dot(X_inv, I-np.mean(I))
#I_corr = I-np.matmul(X,beta)

beta, I_corr = ana.linearize(I,I0,energy)
#I_corr = I-np.matmul(X[:,2:]-np.mean(X[:,2:],axis=0),beta[2:])
plt.figure('I0 correlation corrected')
plt.plot(I0, I_corr, 'o', markersize=0.02)
plt.xlabel('I0')
plt.ylabel('I')
plt.show()





















