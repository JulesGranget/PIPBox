

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
######## STRETCH ########
################################



#tf = tf_conv
def compute_stretch_tf(tf, cond, odor_i, respfeatures_allcond, stretch_point_TF, srate):

    #n_chan = 0
    def stretch_tf_db_n_chan(n_chan):

        tf_stretch_i = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_stretch_i

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))    

    #### verify cycle number
    n_cycles_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[0,:,:], srate)[0].shape[0]

    #### extract
    tf_stretch_allchan = np.zeros((tf.shape[0], n_cycles_stretch, tf.shape[1], stretch_point_TF))

    #n_chan = 0
    for n_chan in range(tf.shape[0]):
        tf_stretch_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_stretch_allchan








################################
######## ALL CONV ########
################################


def precompute_tf_all_conv(sujet):

    #cond = cond_list[0]
    for cond in cond_list:

        os.chdir(os.path.join(path_precompute, 'TF'))
        if os.path.exists(f'{sujet}_tf_conv_{cond}.npy'):
            print(f'{sujet} {cond} ALREADY COMPUTED')
            continue

        print(f'#### {sujet} CONV {cond} ####')

        #### load
        data = load_data_sujet(sujet, cond)
        chan_sel_i = [chan_i for chan_i, chan in enumerate(chan_list_eeg) if chan in chan_list_eeg_short]
        data = data[chan_sel_i,:]

        #### convolution
        wavelets = get_wavelets()

        os.chdir(path_memmap)
        tf_conv = np.memmap(f'{sujet}_{cond}_precompute_convolutions.dat', dtype=np.float32, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
    
        #chan_i = 0
        def compute_tf_convolution_nchan(chan_i):

            print_advancement(chan_i, data.shape[0], steps=[25, 50, 75])

            x = data[chan_i,:]

            tf_i = np.zeros((nfrex, x.shape[0]))

            for fi in range(nfrex):
                
                tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

            tf_conv[chan_i,:,:] = tf_i

        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(chan_i) for chan_i in range(data.shape[0]))

        if debug:
            plt.pcolormesh(tf_conv[0,:,:int(tf_conv.shape[-1]/4)])
            plt.show()

        #### normalize
        print('NORMALIZE')
        tf_conv = norm_tf(sujet, tf_conv, 'zscore')

        if debug:
            tf_plot = tf_conv[0,:,:int(tf_conv.shape[-1]/5)]
            vmin = np.percentile(tf_plot.reshape(-1), 2.5)
            vmax = np.percentile(tf_plot.reshape(-1), 97.5)
            plt.pcolormesh(tf_conv[0,:,:int(tf_conv.shape[-1]/5)], vmin=vmin, vmax=vmax)
            plt.show()

        #### save & transert
        print('SAVE')
        os.chdir(os.path.join(path_precompute, 'TF'))
        np.save(f'{sujet}_tf_conv_{cond}.npy', tf_conv)

        os.chdir(path_memmap)
        try:
            os.remove(f'{sujet}_{cond}_precompute_convolutions.dat')
            del tf_conv
        except:
            pass




















################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #sujet = sujet_list[0]
    for sujet in sujet_list:
    
        # precompute_tf_all_conv(sujet)
        execute_function_in_slurm_bash('n05_precompute_TF', 'precompute_tf_all_conv', [sujet], n_core=15, mem='30G')
        #sync_folders__push_to_crnldata()


