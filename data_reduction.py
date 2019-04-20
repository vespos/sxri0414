import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.signal import savgol_filter as savgol
from scipy.interpolate import interp1d
import IPython
#from importlib import reload

import psana as ps
import ana_fun as ana
reload(ana)

""" ---------------------------- """
exp_name = 'sxri0414'
run = 60
dstr = 'exp={}:run={}:smd'.format(exp_name, run)
dsource = ps.MPIDataSource(dstr)

#h5_dir = '/reg/d/psdm/SXR/sxri0414/hdf5'
h5_dir = './'
fh5 = h5_dir + 'vesp_run60_anaMPI.h5'

smldata = dsource.small_data(fh5, gather_interval=100) #save file

for nevt,evt in enumerate(dsource.events()):
#	if nevt==500:break
	if not (nevt % 100): print('Running event number {}\n'.format(nevt))
	
	#try:
	dl = ps.Detector('SXR:LAS:MCN1:08.RBV')()

	""" Signal """
	det = ps.Detector('acq02')
	res = ana.acquiris(det, evt, debug=0)
#	time, wave_smooth, x, y = ana.acquiris(det, evt, debug=1)
	sig = res[0]['amplitude']
	

	""" I0 """
	I0det = ps.Detector('GMD')
	I0obj = I0det.get(evt)
	try:
#		I0 = I0obj.milliJoulesAverage()
		I0 = I0obj.milliJoulesPerPulse()
	except(AttributeError):
		I0 = np.nan

	""" timig tool """
	ttdet = ps.Detector('TSS_OPAL')
	tt_img = ttdet.raw(evt)
	tt = np.sum(tt_img, 0)

	#except: continue

	#IPython.embed()


	save_data = {
		'dl': dl,
		#'wave': waveform,
		'sig': sig,
		'I0': I0
        'tt': tt
	}
	
	smldata.event(save_data)

smldata.save()













