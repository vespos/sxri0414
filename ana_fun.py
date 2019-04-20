import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import IPython
#from lmfit import Model


def parabola(x,a,x0,offset):
	return a*(x-x0)**2+offset


def gaussian(x, a, x0, sigma):
	return a*exp(-(x-x0)**2/(2*sigma**2))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def fit_peak_parabola_nonlin(waveform, x=None, width_ratio=0.9, plot=False):
    """
    Fit peak to a parabola using nonlinear least square
    
    """
    
    if x is None:
        x = np.arange(waveform.shape[0])

    func = parabola
    
    peak_pos = waveform.argmax()
    peak_width = find_nearest(waveform, width_ratio*waveform[peak_pos])
    peak_width = np.abs(peak_pos-peak_width)
    xr = x[peak_pos-peak_width:peak_pos+peak_width]
    yr = waveform[peak_pos-peak_width:peak_pos+peak_width]

    try:
        p0 = [-1, x[peak_pos], waveform[peak_pos]]
        popt, pcov = curve_fit(func, xr, yr, p0=p0)
        perr = np.sqrt(np.diag(pcov))
    except(ValueError, TypeError, RuntimeError):
        popt = np.array([np.nan, np.nan, np.nan])
        perr = np.array([np.nan, np.nan, np.nan])
        
    if plot:
        plt.figure('Fit')
        plt.plot(x[peak_pos-5*peak_width:peak_pos+5*peak_width], 
                 waveform[peak_pos-5*peak_width:peak_pos+5*peak_width], '.', markersize=7)
        if ~np.isnan(popt[0]):
            xfit = x[peak_pos-peak_width:peak_pos+peak_width]
            yfit = func(xfit, *popt)
        else:
            xfit = x[peak_pos-peak_width:peak_pos+peak_width]
            yfit = func(xfit, *p0)
        plt.plot(xfit, yfit, linewidth=2)
        plt.show()
                
    return popt, perr



def fit_peak_parabola_lin(waveform, x=None, width_ratio=0.9, plot=False):
    """
    Fit peak to a parabola using linear least square
    
    """
        
    if x is None:
        x = np.arange(waveform.shape[0])
    
    peak_pos = waveform.argmax()
    peak_width = find_nearest(waveform, width_ratio*waveform[peak_pos])
    peak_width = np.abs(peak_pos-peak_width)
    xr = x[peak_pos-peak_width:peak_pos+peak_width]
    yr = waveform[peak_pos-peak_width:peak_pos+peak_width]
    
    try:
        X = np.vstack([np.ones(len(xr)), xr, xr**2]).T
        fit = np.linalg.lstsq(X,yr)
        
        """ calculate error on fit parameters """
#        s = np.abs(yr-np.matmul(X,fit))**2 / (X.shape[0]-X.shape[1])
#        cov = np.linalg.inv(np.matmul(X.transpose(), X))
#        fit_err = np.asarray( [s*np.sqrt(cov[ii,ii]) for ii in cov.shape[0]] )
        
        a = fit[0]
        A = a[2]
        x0 = -a[1]/2/a[2]
        offset = a[0]-a[2]*a[1]**2/4/a[2]**2
    except(ValueError, TypeError):
        A = np.nan
        x0 = np.nan
        offset = np.nan
#        fit_err = np.nan
    
    if plot:
        plt.figure('Fit')
        plt.plot(x[peak_pos-5*peak_width:peak_pos+5*peak_width], 
                 waveform[peak_pos-5*peak_width:peak_pos+5*peak_width], '.', markersize=7)
        xfit = x[peak_pos-peak_width:peak_pos+peak_width]
        yfit = parabola(xfit, A,x0,offset)
        plt.plot(xfit, yfit, linewidth=2)
        plt.show()
    
    return np.array([A,x0,offset])
            



def acquiris(detObj, event, channels=0, smooth_window=301, bin_size=30, a0 = -4e10, debug=0):
	""" Analyze the acquiris profile by fitting a parabola to the top 5% of the peak and extracting its peak amplitude. """

	if isinstance(channels, int): channels = [channels]
	results = []
	for channel in channels:
		time = detObj(event)[1][channel,:]
		wave = -detObj(event)[0][channel,:]
		wave_smooth = savgol_filter(wave, smooth_window, 3)
		wave_smooth = wave_smooth - np.mean(wave_smooth[1000:3000])


		""" Fit top of the peak to a parabola """
		func = parabola
		peak_pos = wave_smooth.argmax()
		peak_width = find_nearest(wave_smooth, 0.95*wave_smooth[peak_pos])
		peak_width = np.abs(peak_pos-peak_width)
		x = time[peak_pos-peak_width:peak_pos+peak_width:bin_size]
		y = wave_smooth[peak_pos-peak_width:peak_pos+peak_width:bin_size]

		try:
			popt, pcov = curve_fit(func, x, y, p0=[a0, time[peak_pos], wave_smooth[peak_pos]])
			perr = np.sqrt(np.diag(pcov))
		except:
			popt = np.array([np.nan, np.nan, np.nan])
			perr = np.array([np.nan, np.nan, np.nan])

		results.append({'amplitude': popt[2],
			'amplitude_sig': perr[2]
			})


		if debug:
			plt.figure('channel {}'.format(channel))
			plt.plot(time, wave_smooth)
			if ~np.isnan(popt[0]):
				xfit = time[peak_pos-peak_width:peak_pos+peak_width]
				yfit = func(xfit, *popt)
				plt.plot(xfit, yfit, linewidth=3)
#		IPython.embed()
#	return time, wave_smooth, x, y
	
	return results
#	if len(results)==1: return results[0]
#	else: return results



def linearize_energy(I, I0, energy):
    energy_I0_product = (energy-np.mean(energy))*(I0-np.mean(I0))
    X = np.array([np.ones(len(I0)), I0, energy, energy_I0_product]).transpose()
    X_inv = np.matmul( np.linalg.inv(np.matmul(X.transpose(), X)) , X.transpose() ) # pseudo inverse (Moore-Penrose inverse)
    beta = np.dot(X_inv, I)
    I_corr = np.matmul(X,beta)
    return beta, I_corr



def linearize_slope(I,I0):
    """ get the slope of the I-I0 correlation """
    X = np.array([np.ones(len(I0)), I0]).transpose()
    X_inv = np.linalg.pinv(X)
    beta = np.dot(X_inv,I)
    I_corr = np.matmul(X,beta)
    return beta, I_corr



def linearize_energy_lin(I, I0, energy):
    X = np.array([np.ones(len(I0)), I0, energy]).transpose()
    X_inv = np.linalg.pinv(X) # pseudo inverse (Moore-Penrose inverse)
    beta = np.dot(X_inv, I)
    I_corr = np.matmul(X,beta)
    return beta, I_corr



def get_svd_background(data, return_bkg=False):
    dropshots = data['evr']['code_162'].astype(bool)
    bkgs = data['timeToolOpal'][dropshots, :]
    u,s,v = np.linalg.svd(bkgs)
    if return_bkg: return u,s,v, bkgs
    else: return u,s,v



def subtract_svd_background(data, mask, svd_size=5):
    u,s,v = get_svd_background(data)
    ttdata = data['timeToolOpal']
    """ Fit svd functions to background of each image: """
    fit = np.matmul( ttdata[:,mask], np.linalg.pinv(v[:svd_size,mask]) )
    """ Background: """
    bkgs = np.matmul(fit,v[:svd_size])
    return ttdata - bkgs





