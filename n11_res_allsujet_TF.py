
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False







#########################
######## PLOT TF ########
#########################



#chan = chan_list_eeg_short[0]
def save_tf_allsujet(chan):

    print(f'#### COMPUTE TF STATS {chan} ####', flush=True)

    #### identify if already computed for all
    os.chdir(os.path.join(path_results, 'TF'))

    if os.path.exists(f'allsujet_{chan}.png'):
        print(f'{chan} ALREADY COMPUTED', flush=True)
        return

    #### load data
    data_allcond = {}
    print('#### LOAD BASELINE ####', flush=True)

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

    tf_stretch_baseline_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 46, sujet_list[46]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_baseline_allsujet[sujet_i,:,:] = np.load(f'{sujet}_VS_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    data_allcond['VS'] = np.median(tf_stretch_baseline_allsujet, axis=0)

    print('#### LOAD COND ####', flush=True)

    tf_stretch_cond_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 0, sujet_list[47]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_cond_allsujet[sujet_i,:,:] = np.load(f'{sujet}_CHARGE_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    data_allcond['CHARGE'] = np.median(tf_stretch_cond_allsujet, axis=0)
    
    #### load data thresh
    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH_STATS'))
    
    stats_allcond = {}
    for cond in cond_list:
        stats_allcond[cond] = np.load(f'{chan}_{cond}_allsujet_tf_STATS.npy')

    #### scale    
    vlim = np.abs(np.array([data_allcond['VS'].min(), data_allcond['CHARGE'].min(), data_allcond['VS'].max(), data_allcond['CHARGE'].max()])).max()

    #### plot 
    fig, axs = plt.subplots(ncols=len(cond_list))

    plt.suptitle(f'{chan} tf allsujet count:{len(sujet_list)}')

    fig.set_figheight(5)
    fig.set_figwidth(15)

    #### for plotting l_gamma down
    #c, cond = 1, cond_to_plot[1]
    for c, cond in enumerate(cond_list):

        ax = axs[c]
        ax.set_title(cond, fontweight='bold', rotation=0)

        #### generate time vec
        time_vec = np.arange(stretch_point_ERP)

        #### plot
        ax.pcolormesh(time_vec, frex, data_allcond[cond], vmin=-vlim, vmax=vlim, shading='gouraud', cmap=plt.get_cmap('seismic'))
        # ax.pcolormesh(time_vec, frex, data_allcond[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        #### stats
        ax.contour(time_vec, frex, stats_allcond[cond], levels=0, colors='g')

        ax.vlines(stretch_point_ERP/2, ymin=frex[0], ymax=frex[-1], colors='g')
        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
        ax.set_ylim(frex[0], frex[-1])

    #plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'TF', 'allsujet'))
    fig.savefig(f'{chan}.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()



   
#chan = chan_list_eeg_short[0]
def save_tf_subjectwise(chan):

    print(f'#### COMPUTE TF STATS {chan} ####', flush=True)

    #### identify if already computed for all
    os.chdir(os.path.join(path_results, 'TF'))

    if os.path.exists(f'allsujet_{chan}.png'):
        print(f'{chan} ALREADY COMPUTED', flush=True)
        return

    #### load data
    print('#### LOAD BASELINE ####', flush=True)

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

    tf_stretch_baseline_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 46, sujet_list[46]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_baseline_allsujet[sujet_i,:,:] = np.load(f'{sujet}_VS_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    print('#### LOAD COND ####', flush=True)

    tf_stretch_cond_allsujet = np.zeros((len(sujet_list), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 0, sujet_list[47]
    for sujet_i, sujet in enumerate(sujet_list):

        tf_stretch_cond_allsujet[sujet_i,:,:] = np.load(f'{sujet}_CHARGE_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]    

    data_allcond = {'VS' : tf_stretch_baseline_allsujet, 'CHARGE' : tf_stretch_cond_allsujet}

    #### plot 
    #sujet_i, sujet = 0, sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        fig, axs = plt.subplots(ncols=len(cond_list))

        plt.suptitle(f'{sujet} {chan} tf allsujet count:{len(sujet_list)}')

        fig.set_figheight(5)
        fig.set_figwidth(15)

        #### for plotting l_gamma down
        #c, cond = 1, cond_to_plot[1]
        for c, cond in enumerate(cond_list):

            ax = axs[c]
            ax.set_title(cond, fontweight='bold', rotation=0)

            #### generate time vec
            time_vec = np.arange(stretch_point_ERP)

            #### plot
            # ax.pcolormesh(time_vec, frex, data_allcond[cond], vmin=-vlim, vmax=vlim, shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.pcolormesh(time_vec, frex, data_allcond[cond][sujet_i,:,:], shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.set_yscale('log')

            ax.vlines(stretch_point_ERP/2, ymin=frex[0], ymax=frex[-1], colors='g')
            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
            ax.set_ylim(frex[0], frex[-1])

        #plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'TF', 'subjectwise'))
        fig.savefig(f'{sujet}_{chan}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()


             

    




########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #chan = chan_list_eeg_short[0]
    for chan in chan_list_eeg_short:
                
        save_tf_allsujet(chan)
        save_tf_subjectwise(chan)

        








