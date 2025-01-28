

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import xarray as xr
import joblib

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False






################################
######## ALL CONV ########
################################

#sujet = sujet_list[0]
def precompute_tf_all_conv(sujet):

    #cond = cond_list[0]
    for cond in cond_list:

        os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))
        if os.path.exists(f'{sujet}_{cond}_tf_stretch.npy'):
            print(f'{sujet} {cond} ALREADY COMPUTED')
            continue

        print(f'#### {sujet} CONV {cond} ####')

        #### load
        data = load_data_sujet(sujet, cond)
        chan_sel_i = [chan_i for chan_i, chan in enumerate(chan_list_eeg) if chan in chan_list_eeg_short]
        data = data[chan_sel_i,:]

        #### convolution
        wavelets = get_wavelets()

        tf_conv = np.zeros((data.shape[0], nfrex, data.shape[1]))
    
        #chan_i = 0
        def compute_tf_convolution_nchan(chan_i):

            print_advancement(chan_i, data.shape[0], steps=[25, 50, 75])

            x = data[chan_i,:]

            tf_i = np.zeros((nfrex, x.shape[0]))

            for fi in range(nfrex):
                
                tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

            tf_conv[chan_i,:,:] = tf_i

        joblib.Parallel(n_jobs = n_core, prefer = 'threads')(joblib.delayed(compute_tf_convolution_nchan)(chan_i) for chan_i in range(data.shape[0]))

        if debug:
            plt.pcolormesh(tf_conv[0,:,:int(tf_conv.shape[-1]/4)])
            plt.show()

        #### stretch median
        tf_stretch = np.zeros((len(chan_list_eeg_short), nfrex, stretch_point_ERP))
        respfeatures = load_respfeatures(sujet)[cond]

        for chan_i, chan in enumerate(chan_list_eeg_short):
            tf_stretch[chan_i,:,:] = np.median(stretch_data_tf(respfeatures, stretch_point_ERP, tf_conv[chan_i,:,:], srate)[0], axis=0)

        if debug:
            tf_plot = tf_stretch[0,:,:]
            vmin = np.percentile(tf_plot.reshape(-1), 2.5)
            vmax = np.percentile(tf_plot.reshape(-1), 97.5)
            plt.pcolormesh(tf_plot, vmin=vmin, vmax=vmax)
            plt.show()

        #### normalize
        print('NORMALIZE')
        tf_stretch = norm_tf(sujet, tf_stretch, 'zscore')

        if debug:
            tf_plot = tf_stretch[0,:,:int(tf_stretch.shape[-1]/5)]
            vmin = np.percentile(tf_plot.reshape(-1), 2.5)
            vmax = np.percentile(tf_plot.reshape(-1), 97.5)
            plt.pcolormesh(tf_stretch[0,:,:int(tf_stretch.shape[-1]/5)], vmin=vmin, vmax=vmax)
            plt.show()

            plt.hist(tf_stretch[0,frex_i,:], bins=100)
            plt.show()

            for frex_i in range(nfrex):
                plt.plot(tf_stretch[0,frex_i,:])
            plt.show()

        #### save & transert
        print('SAVE')
        os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))
        np.save(f'{sujet}_{cond}_tf_stretch.npy', tf_stretch)







################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #sujet = sujet_list[0]
    for sujet in sujet_list:
    
        # precompute_tf_all_conv(sujet)
        execute_function_in_slurm_bash('n05_precompute_TF', 'precompute_tf_all_conv', [sujet], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()


