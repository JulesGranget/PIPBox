
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False





################################
######## COMPUTE STATS ########
################################



#chan = chan_list[0]
def precompute_tf_STATS_allsujet(chan):

    print(f'#### COMPUTE TF STATS {chan} ####', flush=True)

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH_STATS'))

    if os.path.exists(f'{chan}_VS_allsujet_tf_STATS.nc') and os.path.exists(f'{chan}_CHARGE_allsujet_tf_STATS.nc'):
        print('ALREADY COMPUTED', flush=True)
        return

    ######## LOAD ########
    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

    print('#### LOAD BASELINE ####', flush=True)

    tf_stretch_baseline_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 0, sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_baseline_allsujet[sujet_i,:,:] = np.load(f'{sujet}_VS_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    print('#### LOAD COND ####', flush=True)

    tf_stretch_cond_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 0, sujet_list[47]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_cond_allsujet[sujet_i,:,:] = np.load(f'{sujet}_CHARGE_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    ######## COMPUTE SURROGATES & STATS ########

    print('COMPUTE SURROGATES', flush=True)
    tf_stats = {}
    tf_stats['VS'], tf_stats['CHARGE'] = get_permutation_cluster_2d(tf_stretch_baseline_allsujet, tf_stretch_cond_allsujet, n_surrogates_tf)
    
    ######## SAVE ########

    print(f'SAVE', flush=True)

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH_STATS'))
    
    for cond in cond_list: 
        np.save(f'{chan}_{cond}_allsujet_tf_STATS.npy', tf_stats[cond])

    





                    
                
            
                

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #chan = chan_list_eeg_short[0]
    for chan in chan_list_eeg_short:
                
        # precompute_tf_STATS_allsujet(chan)
        execute_function_in_slurm_bash('n06_precompute_TF_STATS', 'precompute_tf_STATS_allsujet', [chan], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()
        








