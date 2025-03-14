
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc
from matplotlib.animation import FuncAnimation
import networkx as nx
import matplotlib.patches as patches

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False






########################################
######## SUPPORT FUNCTIONS ########
########################################

#data, pairs = clusters.loc['whole', :].values, clusters['pair'].values
def from_pairs_2mat(data, pairs):

    unique_channels = sorted(set([ch for pair in pairs for ch in pair.split("-")]))
    channel_to_index = {ch: i for i, ch in enumerate(unique_channels)}  # Mapping channel -> index

    mat = np.zeros((len(unique_channels), len(unique_channels)))

    for pair, value in zip(pairs, data):
        ch1, ch2 = pair.split("-") 
        i, j = channel_to_index[ch1], channel_to_index[ch2] 
        mat[i, j] = value
        mat[j, i] = value 

    if debug:

        plt.imshow(mat)
        plt.show()

    return mat











################################
######## PLOT FC ########
################################


def plot_allsujet_FC_time_stretch():

    #fc_metric = 'MI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch', flush=True)

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')        

        cond_sel = ['VS', 'CHARGE']
        fc_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

        for sujet_i, sujet in enumerate(sujet_list_FC):

            _fc_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
            fc_allsujet[sujet_i] = _fc_sujet.values

        fc_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

        fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time_stretch.nc')

        pairs_to_compute = fc_allsujet['pair'].values
        time_vec = fc_allsujet['time'].values

        fc_allsujet = fc_allsujet.median('ntrials')

        fc_allsujet_rscore = fc_allsujet.copy()

        #### rscore
        for cond in cond_sel:

            for pair in pairs_to_compute:

                for sujet in sujet_list_FC:

                    data_chunk = fc_allsujet.loc[sujet, cond, pair, :]
                    fc_allsujet_rscore.loc[sujet, cond, pair, :] = (data_chunk - data_chunk.median('time')) * 0.6745 / scipy.stats.median_abs_deviation(data_chunk)

        if debug:

            pair = pairs_to_compute[0]
            cond = 'CHARGE'
            for sujet_i, sujet in enumerate(sujet_list_FC):
                plt.plot(fc_allsujet_rscore.loc[sujet,cond,pair,:], alpha=0.2)
            plt.plot(fc_allsujet_rscore.loc[:,cond,pair,:].median('sujet'), color='r')
            plt.show()

        ######## IDENTIFY MIN MAX ########

        #### identify min max for allsujet

        vlim_band = {}

        for band in freq_band_fc_list:
        
            vlim = np.array([])

            #pair_i, pair = 0, pairs_to_compute[0]
            for pair_i, pair in enumerate(pairs_to_compute):

                #cond_i, cond = 0, cond_list[0]
                for cond_i, cond in enumerate(cond_list):

                    if fc_metric == 'MI':
                        data_chunk = fc_allsujet_rscore.loc[:, cond, pair, :].median('sujet').values
                        mad = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, cond, pair, :].values, axis=0)
                    else:
                        data_chunk = fc_allsujet_rscore.loc[:, cond, pair, :].median('sujet').values
                        mad = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, cond, pair, :].values, axis=0)
                            
                    data_chunk_up, data_chunk_down = data_chunk + mad, data_chunk - mad
                    vlim = np.concatenate([vlim, data_chunk_up, data_chunk_down])

            vlim = {'min' : vlim.min(), 'max' : vlim.max()}
            vlim_band[band] = vlim

        if debug:

            for pair in pairs_to_compute:

                plt.plot(fc_allsujet_rscore.loc[:, pair, 'CHARGE', :].mean('sujet'))
            
            plt.show()

        n_sujet = fc_allsujet_rscore['sujet'].shape[0]

        for band in freq_band_fc_list:

            #pair_i, pair = 0, pairs_to_compute[0]
            for pair_i, pair in enumerate(pairs_to_compute):

                fig, ax = plt.subplots()

                fig.set_figheight(5)
                fig.set_figwidth(8)

                plt.suptitle(f'stretch {pair} nsujet:{n_sujet}')

                if fc_metric == 'MI':                    

                    cond = fc_allsujet_rscore.loc[:, 'CHARGE', pair, :].median('sujet').values
                    mad_cond = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, 'CHARGE', pair, :].values, axis=0)
                    baseline = fc_allsujet_rscore.loc[:, 'VS', pair, :].median('sujet').values
                    mad_baseline = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, 'VS', pair, :].values, axis=0)

                else:
                
                    cond = fc_allsujet_rscore.loc[:, 'CHARGE', pair, :].median('sujet').values
                    mad_cond = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, 'CHARGE', pair, :].values, axis=0)
                    baseline = fc_allsujet_rscore.loc[:, 'VS', pair, :].median('sujet').values
                    mad_baseline = scipy.stats.median_abs_deviation(fc_allsujet_rscore.loc[:, 'VS', pair, :].values, axis=0)

                ax.set_ylim(vlim_band[band]['min'], vlim_band[band]['max'])

                ax.plot(time_vec, cond, label='CHARGE',color='r')
                ax.fill_between(time_vec, cond+mad_cond, cond-mad_cond, alpha=0.25, color='m')

                ax.plot(time_vec, baseline, label='VS', color='b')
                ax.fill_between(time_vec, baseline+mad_baseline, baseline-mad_baseline, alpha=0.25, color='c')

                if fc_metric == 'MI':
                    _clusters = clusters.loc[pair, :].values
                else:
                    _clusters = clusters.loc[band, pair, :].values
                    
                ax.fill_between(time_vec, vlim_band[band]['min'], vlim_band[band]['max'], where=_clusters.astype('int'), alpha=0.3, color='r')

                ax.vlines(stretch_point_FC/2, ymin=vlim_band[band]['min'], ymax=vlim_band[band]['max'], colors='g')  

                fig.tight_layout()
                plt.legend()

                # plt.show()

                #### save
                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'allpairs'))

                if fc_metric == 'MI':

                    fig.savefig(f'stretch_{pair}.jpeg', dpi=150)

                else:

                    fig.savefig(f'{band}_stretch_{pair}.jpeg', dpi=150)

                fig.clf()
                plt.close('all')
                gc.collect()




# def plot_allsujet_FC_chunk_nostretch():

#     #fc_metric = 'MI'
#     for fc_metric in ['MI', 'ISPC', 'WPLI']:

#         print(f'{fc_metric} PLOT nostretch', flush=True)

#         os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
#         fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
#         clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time.nc')

#         pairs_to_compute = fc_allsujet['pair'].values
#         time_vec = fc_allsujet['time'].values

#         ######## IDENTIFY MIN MAX ########

#         #### identify min max for allsujet

#         vlim_band = {}

#         for band in freq_band_fc_list:
        
#             vlim = np.array([])

#             #pair_i, pair = 0, pairs_to_compute[0]
#             for pair_i, pair in enumerate(pairs_to_compute):

#                 #cond_i, cond = 0, cond_list[0]
#                 for cond_i, cond in enumerate(cond_list):

#                     if fc_metric == 'MI':
#                         data_chunk = fc_allsujet.loc[:, pair, cond, :].mean('sujet').values
#                         sem = fc_allsujet.loc[:, pair, cond, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, cond, :].shape[0])
#                     else:
#                         data_chunk = fc_allsujet.loc[:, band, cond, pair, :].mean('sujet').values
#                         sem = fc_allsujet.loc[:, band, cond, pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, cond, pair, :].shape[0])
                            
#                     data_chunk_up, data_chunk_down = data_chunk + sem, data_chunk - sem
#                     vlim = np.concatenate([vlim, data_chunk_up, data_chunk_down])

#             vlim = {'min' : vlim.min(), 'max' : vlim.max()}
#             vlim_band[band] = vlim

#         if debug:

#             for pair in pairs_to_compute:

#                 plt.plot(fc_allsujet.loc[:, 'CHARGE', pair, :].mean('sujet'))
            
#             plt.show()

#         n_sujet = fc_allsujet['sujet'].shape[0]

#         for band in freq_band_fc_list:

#             #pair_i, pair = 0, pairs_to_compute[0]
#             for pair_i, pair in enumerate(pairs_to_compute):

#                 fig, ax = plt.subplots()

#                 fig.set_figheight(5)
#                 fig.set_figwidth(8)

#                 plt.suptitle(f'{pair} nsujet:{n_sujet}')

#                 if fc_metric == 'MI':
#                     data_chunk = fc_allsujet.loc[:, pair, 'CHARGE', :].mean('sujet').values
#                     sem = fc_allsujet.loc[:, pair, 'CHARGE', :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, 'CHARGE', :].shape[0])
#                     baseline = fc_allsujet.loc[:, pair, 'VS', :].mean('sujet').values
#                     sem_baseline = fc_allsujet.loc[:, pair, 'VS', :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, 'VS', :].shape[0])
#                 else:
#                     data_chunk = fc_allsujet.loc[:, band, 'CHARGE', pair, :].mean('sujet').values
#                     sem = fc_allsujet.loc[:, band, 'CHARGE', pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, 'CHARGE', pair, :].shape[0])
#                     baseline = fc_allsujet.loc[:, band, 'VS', pair, :].mean('sujet').values
#                     sem_baseline = fc_allsujet.loc[:, band, 'VS', pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, 'VS', pair, :].shape[0])

#                 ax.set_ylim(vlim_band[band]['min'], vlim_band[band]['max'])

#                 ax.plot(time_vec, data_chunk, label='CHARGE',color='r')
#                 ax.fill_between(time_vec, data_chunk+sem, data_chunk-sem, alpha=0.25, color='m')

#                 ax.plot(time_vec, baseline, label='VS', color='b')
#                 ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

#                 if fc_metric == 'MI':
#                     _clusters = clusters.loc[pair, :].values
#                 else:
#                     _clusters = clusters.loc[band, pair, :].values
                    
#                 ax.fill_between(time_vec, vlim_band[band]['min'], vlim_band[band]['max'], where=_clusters.astype('int'), alpha=0.3, color='r')

#                 ax.vlines(0, ymin=vlim_band[band]['min'], ymax=vlim_band[band]['max'], colors='g')  

#                 fig.tight_layout()
#                 plt.legend()

#                 # plt.show()

#                 #### save
#                 os.chdir(os.path.join(path_results, 'FC', fc_metric, 'allpairs'))

#                 if fc_metric == 'MI':

#                     fig.savefig(f'nostretch_{pair}.jpeg', dpi=150)

#                 else:

#                     fig.savefig(f'{band}_nostretch_{pair}.jpeg', dpi=150)

#                 fig.clf()
#                 plt.close('all')
#                 gc.collect()





def plot_allsujet_FC_mat_stretch():

    #fc_metric = 'MI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
        clusters_time = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time_stretch.nc')

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')        

        cond_sel = ['VS', 'CHARGE']

        fc_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

        for sujet_i, sujet in enumerate(sujet_list_FC):

            _fc_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
            fc_allsujet[sujet_i] = _fc_sujet.values

        fc_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

        fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

        time_vec = fc_allsujet['time'].values
        phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
        phase_shift = int(stretch_point_FC/4) 
        phase_vec = {'whole' : np.arange(stretch_point_FC), 'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                     'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 
        
        fc_allsujet_median = fc_allsujet.median('ntrials')

        shifted_fc_allsujet = fc_allsujet_median.roll(time=-phase_shift, roll_coords=False)

        #band_i, band = 0, freq_band_fc_list
        for band_i, band in enumerate(freq_band_fc_list):

            fc_mat = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
            fc_mat_mask_signi = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
            fc_mat_only_signi = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

            #phase_i, phase = 0, 'whole'
            for phase_i, phase in enumerate(phase_list):

                if fc_metric == 'MI':
                    fc_mat_mask_signi[phase_i,:,:] = from_pairs_2mat(clusters.loc[phase,:], pairs_to_compute)
                else:
                    fc_mat_mask_signi[phase_i,:,:] = from_pairs_2mat(clusters.loc[phase,band,:], pairs_to_compute)

                #pair_i, pair = 2, pairs_to_compute[2]
                for pair_i, pair in enumerate(pairs_to_compute):

                    A, B = pair.split('-')
                    A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                    if fc_metric == 'MI':
                        data_chunk_diff = shifted_fc_allsujet.loc[:, 'CHARGE', pair, phase_vec[phase]].median('sujet').values - shifted_fc_allsujet.loc[:, 'VS', pair, phase_vec[phase]].median('sujet').values
                        
                    else:
                        data_chunk_diff = shifted_fc_allsujet.loc[:, band, 'CHARGE', pair, phase_vec[phase]].median('sujet').values - shifted_fc_allsujet.loc[:, band, 'VS', pair, phase_vec[phase]].median('sujet').values

                    fc_val = np.median(data_chunk_diff)

                    fc_mat[phase_i, A_i, B_i], fc_mat[phase_i, B_i, A_i] = fc_val, fc_val

                    if fc_metric == 'MI' and clusters.loc[phase,pair].values.astype('bool'):
                        fc_mat_only_signi[phase_i, A_i, B_i], fc_mat_only_signi[phase_i, B_i, A_i] = fc_val, fc_val
                    elif fc_metric != 'MI' and clusters.loc[phase,band,pair].values.astype('bool'):
                        fc_mat_only_signi[phase_i, A_i, B_i], fc_mat_only_signi[phase_i, B_i, A_i] = fc_val, fc_val

            #### plot

            vlim = np.abs((fc_mat.min(), fc_mat.max())).max()

            #fc_type = 'signimat'
            for fc_type in ['fullmat', 'signimat']:

                fig, axs = plt.subplots(ncols=len(phase_list), figsize=(12,5)) 

                for phase_i, phase in enumerate(phase_list):

                    ax = axs[phase_i]

                    if fc_type == 'fullmat':
                        im = ax.imshow(fc_mat[phase_i, :, :], cmap='seismic', vmin=-vlim, vmax=vlim)
                    elif fc_type == 'signimat':
                        im = ax.imshow(fc_mat_only_signi[phase_i, :, :], cmap='seismic', vmin=-vlim, vmax=vlim)
                    ax.set_xticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short, rotation=90)
                    ax.set_xlabel("Electrodes")

                    if phase_i == 0:
                        ax.set_yticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short)
                        ax.set_ylabel("Electrodes")

                    if fc_type == 'fullmat':
                        _fc_mat_mask_signi = fc_mat_mask_signi[phase_i,:,:]

                        for i in range(_fc_mat_mask_signi.shape[0]):
                            for j in range(_fc_mat_mask_signi.shape[1]):
                                if _fc_mat_mask_signi[i, j]:
                                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=4, edgecolor='g', facecolor='none')
                                    ax.add_patch(rect)

                    ax.set_title(phase)

                fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity Strength")

                if fc_metric == 'MI':
                    plt.suptitle("MI FC DIFF")
                else:
                    plt.suptitle(f'{fc_metric} {band} FC')

                # plt.show()

                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                if fc_metric == 'MI':
                    plt.savefig(f'stretch_MI_FC_{fc_type}.jpeg', dpi=150)

                else:
                    plt.savefig(f'stretch_{fc_metric}_{band}_FC_{fc_type}.jpeg', dpi=150)
                    
                if fc_type == "signimat":
                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot', 'summary'))
                    if fc_metric == 'MI':
                        plt.savefig(f'stretch_MI_FC_{fc_type}.jpeg', dpi=150)
                    else:
                        plt.savefig(f'stretch_{fc_metric}_{band}_FC_{fc_type}.jpeg', dpi=150)

                plt.close('all')
                gc.collect()

            def get_visu_data_fullmat(time_window):

                start_window = np.arange(0, time_vec.size, time_window)
                n_times = np.arange(start_window.size)
                
                cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
                data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

                #win_i, win_start = 15, start_window[15]
                for win_i, win_start in enumerate(start_window):

                    if win_start == start_window[-1]:
                        continue
                    
                    win_stop = start_window[win_i+1]

                    _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):
                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
                        cond_i, baseline_i = np.where(shifted_fc_allsujet['cond'].values == 'CHARGE')[0][0], np.where(shifted_fc_allsujet['cond'].values == 'VS')[0][0] 

                        if fc_metric == 'MI':
                            data_chunk_diff = shifted_fc_allsujet[:, cond_i, pair_i, win_start:win_stop].median('sujet').values - shifted_fc_allsujet[:, baseline_i, pair_i, win_start:win_stop].median('sujet').values
                            _clusters = clusters_time[pair_i, win_start:win_stop].values
                        else:
                            data_chunk_diff = shifted_fc_allsujet[:, cond_i, band_i, pair_i, win_start:win_stop].median('sujet').values - shifted_fc_allsujet[:, cond_i, band_i, pair_i, win_start:win_stop].median('sujet').values
                            _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

                        fc_val = np.median(data_chunk_diff)
                        _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                        
                        cluster_val = _clusters.sum() > 0
                        _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

                    data_chunk_wins[win_i, :,:] = _fc_mat
                    cluster_mask_wins[win_i, :,:] = _cluster_mat

                return n_times, start_window, data_chunk_wins, cluster_mask_wins
            
            def get_visu_data_matsigni(time_window):

                start_window = np.arange(0, time_vec.size, time_window)
                n_times = np.arange(start_window.size)
                
                cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
                data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

                #win_i, win_start = 23, start_window[23]
                for win_i, win_start in enumerate(start_window):

                    if win_start == start_window[-1]:
                        continue
                    
                    win_stop = start_window[win_i+1]

                    _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):
                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
                        cond_i, baseline_i = np.where(shifted_fc_allsujet['cond'].values == 'CHARGE')[0][0], np.where(shifted_fc_allsujet['cond'].values == 'VS')[0][0] 

                        if fc_metric == 'MI':
                            data_chunk_diff = shifted_fc_allsujet[:, cond_i, pair_i, win_start:win_stop].median('sujet').values - shifted_fc_allsujet[:, baseline_i, pair_i, win_start:win_stop].median('sujet').values
                            _clusters = clusters_time[pair_i, win_start:win_stop].values
                        else:
                            data_chunk_diff = shifted_fc_allsujet[:, band_i, cond_i, pair_i, win_start:win_stop].median('sujet').values - shifted_fc_allsujet[:, band_i, baseline_i, pair_i, win_start:win_stop].median('sujet').values
                            _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

                        fc_val = np.median(data_chunk_diff)
                        if _clusters.sum() > 0:
                            _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                        
                        cluster_val = _clusters.sum() > 0
                        _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

                    data_chunk_wins[win_i, :,:] = _fc_mat
                    cluster_mask_wins[win_i, :,:] = _cluster_mat

                return n_times, start_window, data_chunk_wins, cluster_mask_wins
            
            def update_fullmat(frame):
                    
                ax.clear()

                ax.set_title(f'{np.round(time_vec[start_window[frame]], 5)}')
                
                ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

                for i in range(chan_list_eeg_short.shape[0]):
                    for j in range(chan_list_eeg_short.shape[0]):
                        if cluster_mask_wins[frame, i, j]:
                            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            edgecolor='green', facecolor='none', lw=4)
                            ax.add_patch(rect)

                phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][1]
                ax.set_xlabel("Electrodes")
                ax.set_ylabel("Electrodes")
                ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")

                return [ax]
            
            def update_matsigni(frame):
                    
                ax.clear()
                
                ax.set_title(f'{np.round(time_vec[start_window[frame]], 5)}')
                
                ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

                ax.set_xticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short, rotation=90)
                ax.set_yticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short)

                phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][1]
                ax.set_xlabel("Electrodes")
                ax.set_ylabel("Electrodes")
                ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")

                return [ax]
            
            #fc_type = 'matsigni'
            for fc_type in ['fullmat', 'matsigni']:

                time_window = 10
                if fc_type == 'fullmat':
                    n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_fullmat(time_window)
                elif fc_type == 'matsigni':
                    n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_matsigni(time_window)
                vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

                if debug:

                    win_i = 10

                    plt.matshow(data_chunk_wins[win_i,:,:], cmap='seismic')

                    for i in range(chan_list_eeg_short.shape[0]):
                        for j in range(chan_list_eeg_short.shape[0]):
                            if cluster_mask_wins[win_i, i, j]:
                                plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))

                    plt.colorbar(label='Connectivity Strength')
                    plt.clim(-vlim, vlim)
                    plt.xticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
                    plt.yticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short)
                    plt.xlabel("Electrodes")
                    plt.ylabel("Electrodes")
                    if fc_metric == 'MI':
                        plt.title(f"MI FC start{np.round(time_vec[start_window[win_i]], 2)}")
                    else:
                        plt.title(f"{fc_metric} {band} FC start{np.round(time_vec[start_window[win_i]], 2)}")
                    plt.show()

                # Create topoplot frames
                fig, ax = plt.subplots()
                cax = ax.matshow(np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short))), cmap='seismic', vmin=-vlim, vmax=vlim)
                colorbar = plt.colorbar(cax, ax=ax)

                # Animation
                if fc_type == 'fullmat':
                    ani = FuncAnimation(fig, update_fullmat, frames=n_times, interval=1000)
                elif fc_type == 'matsigni':
                    ani = FuncAnimation(fig, update_matsigni, frames=n_times, interval=1000)
                # plt.show()

                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                
                if fc_metric == 'MI':
                    ani.save(f"stretch_{fc_metric}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")
                else:
                    ani.save(f"stretch_{fc_metric}_{band}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")  
                

                if fc_type == 'matsigni':

                    time_window = 10
                    n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_matsigni(time_window)
                    vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

                    # EEG 10-20 system positions (approximate, normalized for plotting)
                    eeg_positions = {
                        'C3': (-0.5, 0.3), 'C4': (0.5, 0.3),
                        'CP1': (-0.3, 0.0), 'CP2': (0.3, 0.0),
                        'Cz': (0.0, 0.3), 'F3': (-0.5, 0.7),
                        'F4': (0.5, 0.7), 'FC1': (-0.3, 0.5),
                        'FC2': (0.3, 0.5), 'Fz': (0.0, 0.7)
                    }

                    # Create graph
                    G = nx.Graph()
                    G.add_nodes_from(eeg_positions.keys())

                    # Define the update function for animation
                    def update_graph(frame):
                        plt.clf()
                        ax = plt.gca()

                        phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][1]
                        ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")
                        
                        # Clear and add new edges
                        G.clear()
                        G.add_nodes_from(eeg_positions.keys())
                        
                        mat_connectivity = data_chunk_wins[frame,:,:]
                        
                        for i, ch1 in enumerate(eeg_positions.keys()):
                            for j, ch2 in enumerate(eeg_positions.keys()):
                                if i < j:  # Avoid duplicate edges
                                    weight = mat_connectivity[i, j]
                                    color = 'red' if weight > 0 else 'blue'
                                    G.add_edge(ch1, ch2, weight=abs(weight) * 5, color=color)
                        
                        # Draw the graph
                        edges = G.edges()
                        edge_colors = [G[u][v]['color'] for u, v in edges]
                        edge_widths = [G[u][v]['weight'] for u, v in edges]
                        
                        nx.draw(G, eeg_positions, with_labels=True, node_size=500, font_size=8, edge_color=edge_colors, width=edge_widths, ax=ax)
                        
                        # Add colorbar
                        sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-vlim, vmax=vlim))
                        sm.set_array([])
                        plt.colorbar(sm, ax=ax, label='Connectivity Strength')

                    # Create animation
                    fig = plt.figure()
                    ani = FuncAnimation(fig, update_graph, frames=n_times, interval=1000)
                    # plt.show()

                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                
                    if fc_metric == 'MI':
                        ani.save(f"stretch_{fc_metric}_FC_graph_animation_allsujet_{fc_type}.gif", writer="pillow")
                    else:
                        ani.save(f"stretch_{fc_metric}_{band}_FC_graph_animation_allsujet_{fc_type}.gif", writer="pillow")  





# def plot_allsujet_FC_mat_nostretch():

#     #fc_metric = 'MI'
#     for fc_metric in ['MI', 'ISPC', 'WPLI']:

#         print(f'{fc_metric} PLOT nostretch', flush=True)

#         os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
#         fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
#         clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state.nc')
#         clusters_time = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time_stretch.nc')

#         pairs_to_compute = fc_allsujet['pair'].values
#         time_vec = fc_allsujet['time'].values
#         phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
#         phase_shift = 125 
#         # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
#         phase_vec = {'whole' : np.arange(stretch_point_ERP), 'I' : np.arange(250), 'T_IE' : np.arange(250)+250, 'E' : np.arange(250)+500, 'T_EI' : np.arange(250)+750} 

#         shifted_fc_allsujet = fc_allsujet.roll(time=-phase_shift, roll_coords=False)

#         #band_i, band = 0, freq_band_fc_list
#         for band_i, band in enumerate(freq_band_fc_list): 

#             fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
#             fc_mat_mask_signi = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
#             fc_mat_only_signi = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

#             if fc_metric == 'MI':
#                 fc_mat_mask_signi[:,:] = from_pairs_2mat(clusters, pairs_to_compute)
#             else:
#                 fc_mat_mask_signi[:,:] = from_pairs_2mat(clusters.loc[band,:], pairs_to_compute)

#             #pair_i, pair = 2, pairs_to_compute[2]
#             for pair_i, pair in enumerate(pairs_to_compute):

#                 A, B = pair.split('-')
#                 A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

#                 if fc_metric == 'MI':
#                     data_chunk_diff = fc_allsujet.loc[:, pair, 'CHARGE', :].mean('sujet').values - fc_allsujet.loc[:, pair, 'VS', :].mean('sujet').values
                    
#                 else:
#                     data_chunk_diff = fc_allsujet.loc[:, band, 'CHARGE', pair, :].mean('sujet').values - fc_allsujet.loc[:, band, 'VS', pair, :].mean('sujet').values

#                 fc_val = data_chunk_diff.mean()

#                 fc_mat[A_i, B_i], fc_mat[B_i, A_i] = fc_val, fc_val

#                 if fc_mat_mask_signi[A_i, B_i].astype('bool'):
#                     fc_mat_only_signi[A_i, B_i], fc_mat_only_signi[B_i, A_i] = fc_val, fc_val

#             #### plot

#             vlim = np.abs((fc_mat.min(), fc_mat.max())).max()

#             for fc_type in ['fullmat', 'signimat']:

#                 fig, ax = plt.subplots() 

#                 if fc_type == 'fullmat':
#                     im = ax.imshow(fc_mat, cmap='seismic', vmin=-vlim, vmax=vlim)
#                 elif fc_type == 'signimat':
#                     im = ax.imshow(fc_mat_only_signi, cmap='seismic', vmin=-vlim, vmax=vlim)

#                 ax.set_xticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short, rotation=90)
#                 ax.set_xlabel("Electrodes")

#                 ax.set_yticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short)
#                 ax.set_ylabel("Electrodes")

#                 if fc_type == 'fullmat':

#                     for i in range(fc_mat_mask_signi.shape[0]):
#                         for j in range(fc_mat_mask_signi.shape[1]):
#                             if fc_mat_mask_signi[i, j]:
#                                 rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=4, edgecolor='g', facecolor='none')
#                                 ax.add_patch(rect)

#                 fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity Strength")

#                 if fc_metric == 'MI':
#                     plt.suptitle("MI FC")
#                 else:
#                     plt.suptitle(f'{fc_metric} {band} FC')

#                 # plt.show()

#                 #### both save
#                 os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

#                 if fc_metric == 'MI':
#                     plt.savefig(f'nostretch_MI_FC_{fc_type}.jpeg', dpi=150)

#                 else:
#                     plt.savefig(f'nostretch_{fc_metric}_{band}_FC_{fc_type}.jpeg', dpi=150)

#                 plt.close('all')
#                 gc.collect()

#             def get_visu_data_fullmat(time_window):

#                 time_chunk_points = int(time_window * srate)
#                 start_window = np.arange(0, time_vec.size, time_chunk_points)
#                 start_window_sec = time_vec[start_window]
#                 n_times = np.arange(start_window.size)
                
#                 cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                 data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

#                 #win_i, win_start = 15, start_window[15]
#                 for win_i, win_start in enumerate(start_window):

#                     if win_start == start_window[-1]:
#                         continue
                    
#                     win_stop = start_window[win_i+1]

#                     _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                     _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

#                     #pair_i, pair = 2, pairs_to_compute[2]
#                     for pair_i, pair in enumerate(pairs_to_compute):
#                         A, B = pair.split('-')
#                         A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
#                         cond_i, baseline_i = np.where(fc_allsujet['cond'].values == 'CHARGE')[0][0], np.where(fc_allsujet['cond'].values == 'VS')[0][0] 

#                         if fc_metric == 'MI':
#                             data_chunk_diff = fc_allsujet[:, pair_i, cond_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, pair_i, baseline_i, win_start:win_stop].mean('sujet').values
#                             _clusters = clusters_time[pair_i, win_start:win_stop].values
#                         else:
#                             data_chunk_diff = fc_allsujet[:, band_i, cond_i, pair_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, band_i, baseline_i, pair_i, win_start:win_stop].mean('sujet').values
#                             _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

#                         fc_val = data_chunk_diff.mean()
#                         _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                        
#                         cluster_val = _clusters.sum() > 0
#                         _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

#                     data_chunk_wins[win_i, :,:] = _fc_mat
#                     cluster_mask_wins[win_i, :,:] = _cluster_mat

#                 return n_times, start_window, data_chunk_wins, cluster_mask_wins
            
#             def get_visu_data_matsigni(time_window):

#                 time_chunk_points = int(time_window * srate)
#                 start_window = np.arange(0, time_vec.size, time_chunk_points)
#                 start_window_sec = time_vec[start_window]
#                 n_times = np.arange(start_window.size)
                
#                 cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                 data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

#                 #win_i, win_start = 15, start_window[15]
#                 for win_i, win_start in enumerate(start_window):

#                     if win_start == start_window[-1]:
#                         continue
                    
#                     win_stop = start_window[win_i+1]

#                     _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                     _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

#                     #pair_i, pair = 2, pairs_to_compute[2]
#                     for pair_i, pair in enumerate(pairs_to_compute):
#                         A, B = pair.split('-')
#                         A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
#                         cond_i, baseline_i = np.where(fc_allsujet['cond'].values == 'CHARGE')[0][0], np.where(fc_allsujet['cond'].values == 'VS')[0][0] 

#                         if fc_metric == 'MI':
#                             data_chunk_diff = fc_allsujet[:, pair_i, cond_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, pair_i, baseline_i, win_start:win_stop].mean('sujet').values
#                             _clusters = clusters_time[pair_i, win_start:win_stop].values
#                         else:
#                             data_chunk_diff = fc_allsujet[:, band_i, cond_i, pair_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, band_i, baseline_i, pair_i, win_start:win_stop].mean('sujet').values
#                             _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

#                         fc_val = data_chunk_diff.mean()
#                         if _clusters.sum() > 0:
#                             _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                        
#                         cluster_val = _clusters.sum() > 0
#                         _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

#                     data_chunk_wins[win_i, :,:] = _fc_mat
#                     cluster_mask_wins[win_i, :,:] = _cluster_mat

#                 return n_times, start_window, data_chunk_wins, cluster_mask_wins
            
#             def update_fullmat(frame):
                    
#                     ax.clear()
#                     ax.set_title(np.round(time_vec[start_window[frame]], 5))
                    
#                     ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

#                     for i in range(chan_list_eeg_short.shape[0]):
#                         for j in range(chan_list_eeg_short.shape[0]):
#                             if cluster_mask_wins[frame, i, j]:
#                                 rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
#                                                 edgecolor='green', facecolor='none', lw=4)
#                                 ax.add_patch(rect)

#                     ax.set_xticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short, rotation=90)
#                     ax.set_yticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short)
#                     ax.set_xlabel("Electrodes")
#                     ax.set_ylabel("Electrodes")
#                     ax.set_title(f"MI FC start{np.round(time_vec[start_window[frame]], 2)}")

#                     return [ax]
            
#             def update_matsigni(frame):
                    
#                     ax.clear()
#                     ax.set_title(np.round(time_vec[start_window[frame]], 5))
                    
#                     ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

#                     ax.set_xticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short, rotation=90)
#                     ax.set_yticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short)
#                     ax.set_xlabel("Electrodes")
#                     ax.set_ylabel("Electrodes")
#                     ax.set_title(f"MI FC start{np.round(time_vec[start_window[frame]], 2)}")

#                     return [ax]
            
#             for fc_type in ['fullmat', 'matsigni']:

#                 time_window = 0.1#in s
#                 if fc_type == 'fullmat':
#                     n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_fullmat(time_window)
#                 elif fc_type == 'matsigni':
#                     n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_matsigni(time_window)
#                 vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

#                 if debug:

#                     win_i = 15

#                     plt.matshow(data_chunk_wins[win_i,:,:], cmap='seismic')

#                     for i in range(chan_list_eeg_short.shape[0]):
#                         for j in range(chan_list_eeg_short.shape[0]):
#                             if cluster_mask_wins[win_i, i, j]:
#                                 plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))

#                     plt.colorbar(label='Connectivity Strength')
#                     plt.clim(-vlim, vlim)
#                     plt.xticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
#                     plt.yticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short)
#                     plt.xlabel("Electrodes")
#                     plt.ylabel("Electrodes")
#                     if fc_metric == 'MI':
#                         plt.title(f"MI FC start{np.round(time_vec[start_window[win_i]], 2)}")
#                     else:
#                         plt.title(f"{fc_metric} {band} FC start{np.round(time_vec[start_window[win_i]], 2)}")
#                     plt.show()

#                 # Create topoplot frames
#                 fig, ax = plt.subplots()
#                 cax = ax.matshow(np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short))), cmap='seismic', vmin=-vlim, vmax=vlim)
#                 colorbar = plt.colorbar(cax, ax=ax)

#                 # Animation
#                 if fc_type == 'fullmat':
#                     ani = FuncAnimation(fig, update_fullmat, frames=n_times, interval=1000)
#                 elif fc_type == 'matsigni':
#                     ani = FuncAnimation(fig, update_matsigni, frames=n_times, interval=1000)
#                 # plt.show()

#                 os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                
#                 if fc_metric == 'MI':
#                     ani.save(f"nostretch_{fc_metric}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")
#                 else:
#                     ani.save(f"nostretch_{fc_metric}_{band}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")  






def plot_allsujet_FC_graph_stretch():

    #fc_metric = 'MI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
        clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')        

        cond_sel = ['VS', 'CHARGE']

        fc_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

        for sujet_i, sujet in enumerate(sujet_list_FC):

            _fc_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
            fc_allsujet[sujet_i] = _fc_sujet.values

        fc_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

        fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

        fc_allsujet_median = fc_allsujet.median('ntrials')

        time_vec = fc_allsujet['time'].values
        phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
        phase_shift = int(stretch_point_FC/4) 
        phase_vec = {'whole' : np.arange(stretch_point_FC), 'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                     'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 
        
        #band_i, band = 0, freq_band_fc_list
        for band_i, band in enumerate(freq_band_fc_list):

            fc_mat = np.zeros((len(phase_list), len(cond_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
            fc_mat_mask_signi = np.zeros((len(phase_list), len(cond_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
            fc_mat_only_signi = np.zeros((len(phase_list), len(cond_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

            for cond_i, cond in enumerate(cond_list):

                #phase_i, phase = 0, 'whole'
                for phase_i, phase in enumerate(phase_list):

                    if fc_metric == 'MI':
                        fc_mat_mask_signi[phase_i,cond_i,:,:] = from_pairs_2mat(clusters.loc[phase,:], pairs_to_compute)
                    else:
                        fc_mat_mask_signi[phase_i,cond_i,:,:] = from_pairs_2mat(clusters.loc[phase,band,:], pairs_to_compute)

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                        if fc_metric == 'MI':
                            data_chunk = fc_allsujet_median.loc[:, cond, pair, phase_vec[phase]].median('sujet').values
                            
                        else:
                            data_chunk = fc_allsujet_median.loc[:, band, cond, pair, phase_vec[phase]].median('sujet').values

                        fc_val = np.median(data_chunk)

                        fc_mat[phase_i, cond_i, A_i, B_i], fc_mat[phase_i, cond_i, B_i, A_i] = fc_val, fc_val

                        if fc_metric == 'MI' and clusters.loc[phase,pair].values.astype('bool'):
                            fc_mat_only_signi[phase_i, cond_i, A_i, B_i], fc_mat_only_signi[phase_i, cond_i, B_i, A_i] = fc_val, fc_val
                        elif fc_metric != 'MI' and clusters.loc[phase,band,pair].values.astype('bool'):
                            fc_mat_only_signi[phase_i, cond_i, A_i, B_i], fc_mat_only_signi[phase_i, cond_i, B_i, A_i] = fc_val, fc_val

            if debug:

                plt.imshow(fc_mat[0,0,:,:])
                plt.show()

                plt.imshow(fc_mat_only_signi[0,0,:,:])
                plt.show()

                plt.imshow(fc_mat_mask_signi[0,0,:,:])
                plt.show()

            #mat, mask_graph_metric = fc_mat[0, 0, :, :], fc_mat_mask_signi[0,0,:,:]
            def thresh_fc_mat(mat, mode='mask', percentile_graph_metric=50, mask_graph_metric=None):

                if mode == 'percentile':

                    mat_thresh = mat.copy()

                    mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                    
                    if debug:
                        np.sum(mat_values > np.percentile(mat_values, 90))

                        count, bin, fig = plt.hist(mat_values)
                        plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                        plt.vlines(np.percentile(mat_values, 75), ymin=count.min(), ymax=count.max(), color='r')
                        plt.show()

                    #### apply thresh
                    for chan_i in range(mat.shape[0]):
                        mat_thresh[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

                if mode == 'mask':

                    mat_thresh = mat * mask_graph_metric
                if mat_thresh.sum() == 0:

                    return mat_thresh

                #### verify that the graph is fully connected
                chan_i_to_remove = []
                for chan_i in range(mat_thresh.shape[0]):
                    if np.sum(mat_thresh[chan_i,:]) == 0:
                        chan_i_to_remove.append(chan_i)

                mat_thresh_i_mask = [i for i in range(mat_thresh.shape[0]) if i not in chan_i_to_remove]

                if len(chan_i_to_remove) != 0:
                    for row in range(2):
                        if row == 0:
                            mat_thresh = mat_thresh[mat_thresh_i_mask,:]
                        elif row == 1:
                            mat_thresh = mat_thresh[:,mat_thresh_i_mask]

                if debug:
                    plt.imshow(mat_thresh)
                    plt.show()

                return mat_thresh
            
            df_graph_metrics = pd.DataFrame()

            for mode in ['percentile', 'mask']:

                for sujet_i, sujet in enumerate(sujet_list_FC):

                    for cond_i, cond in enumerate(cond_list):

                        for phase_i, phase in enumerate(phase_list):

                            if fc_mat_mask_signi[phase_i,cond_i,:,:].sum() == 0:

                                _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [0], 'betweenness' : [0], 
                                            'closeness' : [0], 'hubs' : [0],
                                            'clustering_coeff' : [0], 'local_efficiency' : [0]})

                                df_graph_metrics = pd.concat((df_graph_metrics, _df))

                            else:

                                # Create a graph from the adjacency matrix
                                mat = thresh_fc_mat(fc_mat[phase_i,cond_i,:,:], mode=mode, percentile_graph_metric=50 ,mask_graph_metric=fc_mat_mask_signi[phase_i,cond_i,:,:])
                                graph = nx.from_numpy_array(mat)
                                nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

                                if debug:
                                    # Plot the graph
                                    plt.figure(figsize=(10, 8))
                                    pos = nx.spring_layout(graph)  # Layout for visualization
                                    nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
                                    plt.title("Graph Visualization")
                                    plt.show()

                                #### metrics
                                degree = nx.degree_centrality(graph)
                                # Formula: degree_centrality(v) = degree(v) / (n - 1), where n is the number of nodes

                                betweenness = nx.betweenness_centrality(graph)
                                # Formula: betweenness_centrality(v) = sum of (shortest paths through v / total shortest paths)

                                closeness = nx.closeness_centrality(graph)
                                # Formula: closeness_centrality(v) = 1 / (sum of shortest path distances from v to all other nodes)

                                clustering_coeff = nx.clustering(graph)

                                local_efficiency = nx.local_efficiency(graph)

                                # Hubness (using HITS algorithm)
                                hubs, authorities = nx.hits(graph)
                                # Formula: HITS hub score: importance of a node as a hub, based on linking to authorities

                                for chan in chan_list_eeg_short:

                                    try:
                                        _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [degree[chan]], 'betweenness' : [betweenness[chan]], 
                                                'closeness' : [closeness[chan]], 'hubs' : [hubs[chan]],
                                                'clustering_coeff' : [clustering_coeff[chan]], 'local_efficiency' : [local_efficiency]})

                                        df_graph_metrics = pd.concat((df_graph_metrics, _df))

                                    except:

                                        _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [0], 'betweenness' : [0], 
                                                'closeness' : [0], 'hubs' : [0],
                                                'clustering_coeff' : [0], 'local_efficiency' : [0]})

                                        df_graph_metrics = pd.concat((df_graph_metrics, _df))

            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

            #metric = 'degree'
            for metric in ['degree', 'betweenness', 'closeness', 'hubs', 'clustering_coeff', 'local_efficiency']:

                for mode in ['mask', 'percentile']: 

                    fig, axs = plt.subplots(ncols=len(phase_list), figsize=(15, 5), sharey=True)

                    for phase_i, phase in enumerate(phase_list):

                        ax = axs[phase_i]
                        df_plot = df_graph_metrics.query(f"mode == '{mode}' and phase =='{phase}'")
                        sns.barplot(data=df_plot, x="chan", y=metric, hue="cond", alpha=0.6, ax=ax)
                        ax.set_title(phase) 
                        ax.set_xlabel("Channel")
                        if phase_i == 0:
                            ax.set_ylabel(metric)
                        else:
                            ax.set_ylabel("")

                        if debug:

                            df_plot = df_graph_metrics.query(f"mode == '{mode}' and cond =='VS'")
                            fig, ax = plt.subplots()
                            sns.barplot(data=df_plot, x="chan", y=metric, hue='phase', alpha=0.6, ax=ax)
                            plt.show()

                    if fc_metric == 'MI':
                        plt.suptitle(f"{metric} {mode}")
                    else :
                        plt.suptitle(f"{metric} {band} {mode}")

                    # plt.show()

                if fc_metric == 'MI':
                    plt.savefig(f"stretch_{fc_metric}_graph_{metric}.png")
                else:
                    plt.savefig(f"stretch_{fc_metric}_{band}_graph_{metric}.png")

                plt.close("all")




# def plot_allsujet_FC_graph_nostretch():

#     stretch=False

#     #fc_metric = 'MI'
#     for fc_metric in ['MI', 'ISPC', 'WPLI']:

#         print(f'{fc_metric} PLOT stretch:{stretch}', flush=True)

#         os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
#         if stretch:
#             fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
#             clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
#             clusters_time = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time_stretch.nc')
#         else:
#             fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
#             clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state.nc')
#             clusters_time = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_time_stretch.nc')

#         pairs_to_compute = fc_allsujet['pair'].values
#         time_vec = fc_allsujet['time'].values
#         phase_list = ['whole', 'inspi', 'expi']
#         phase_vec = {'whole' : time_vec, 'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

#         #band_i, band = 0, freq_band_fc_list
#         for band_i, band in enumerate(freq_band_fc_list):

#             #### stretch compute
#             if stretch:

#                 fc_mat = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                 fc_mat_mask_signi = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
#                 fc_mat_only_signi = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

#                 #phase_i, phase = 0, 'whole'
#                 for phase_i, phase in enumerate(phase_list):

#                     if fc_metric == 'MI':
#                         fc_mat_mask_signi[phase_i,:,:] = from_pairs_2mat(clusters.loc[phase,:], pairs_to_compute)
#                     else:
#                         fc_mat_mask_signi[phase_i,:,:] = from_pairs_2mat(clusters.loc[phase,band,:], pairs_to_compute)

#                     #pair_i, pair = 2, pairs_to_compute[2]
#                     for pair_i, pair in enumerate(pairs_to_compute):

#                         A, B = pair.split('-')
#                         A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

#                         if fc_metric == 'MI':
#                             data_chunk_diff = fc_allsujet.loc[:, pair, 'CHARGE', phase_vec[phase]].mean('sujet').values - fc_allsujet.loc[:, pair, 'VS', phase_vec[phase]].mean('sujet').values
                            
#                         else:
#                             data_chunk_diff = fc_allsujet.loc[:, band, 'CHARGE', pair, phase_vec[phase]].mean('sujet').values - fc_allsujet.loc[:, band, 'VS', pair, phase_vec[phase]].mean('sujet').values

#                         fc_val = data_chunk_diff.mean()

#                         fc_mat[phase_i, A_i, B_i], fc_mat[phase_i, B_i, A_i] = fc_val, fc_val

#                         if fc_metric == 'MI' and clusters.loc[phase,pair].values.astype('bool'):
#                             fc_mat_only_signi[phase_i, A_i, B_i], fc_mat_only_signi[phase_i, B_i, A_i] = fc_val, fc_val
#                         elif fc_metric != 'MI' and clusters.loc[phase,band,pair].values.astype('bool'):
#                             fc_mat_only_signi[phase_i, A_i, B_i], fc_mat_only_signi[phase_i, B_i, A_i] = fc_val, fc_val

#             #mat = fc_mat_cond[0]
#             def thresh_fc_mat(mat, percentile_graph_metric=50):

#                 mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                
#                 if debug:
#                     np.sum(mat_values > np.percentile(mat_values, 90))

#                     count, bin, fig = plt.hist(mat_values)
#                     plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
#                     plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
#                     plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
#                     plt.vlines(np.percentile(mat_values, 75), ymin=count.min(), ymax=count.max(), color='r')
#                     plt.show()

#                 #### apply thresh
#                 for chan_i in range(mat.shape[0]):
#                     mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

#                 #### verify that the graph is fully connected
#                 chan_i_to_remove = []
#                 for chan_i in range(mat.shape[0]):
#                     if np.sum(mat[chan_i,:]) == 0:
#                         chan_i_to_remove.append(chan_i)

#                 mat_i_mask = [i for i in range(mat.shape[0]) if i not in chan_i_to_remove]

#                 if len(chan_i_to_remove) != 0:
#                     for row in range(2):
#                         if row == 0:
#                             mat = mat[mat_i_mask,:]
#                         elif row == 1:
#                             mat = mat[:,mat_i_mask]

#                 if debug:
#                     plt.matshow(mat)
#                     plt.show()

#                 return mat
            
#             df_graph_metrics = pd.DataFrame()

#             for sujet in sujet_list:

#                 for cond in cond_list:

#                     # Create a graph from the adjacency matrix
#                     mat = thresh_fc_mat(fc_mat_cond.loc[sujet,cond,:,:].values)
#                     graph = nx.from_numpy_array(mat)
#                     nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

#                     if debug:
#                         # Plot the graph
#                         plt.figure(figsize=(10, 8))
#                         pos = nx.spring_layout(graph)  # Layout for visualization
#                         nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
#                         plt.title("Graph Visualization")
#                         plt.show()


#                     #### metrics
#                     degree = nx.degree_centrality(graph)
#                     # Formula: degree_centrality(v) = degree(v) / (n - 1), where n is the number of nodes

#                     betweenness = nx.betweenness_centrality(graph)
#                     # Formula: betweenness_centrality(v) = sum of (shortest paths through v / total shortest paths)

#                     closeness = nx.closeness_centrality(graph)
#                     # Formula: closeness_centrality(v) = 1 / (sum of shortest path distances from v to all other nodes)

#                     clustering_coeff = nx.clustering(graph)

#                     local_efficiency = nx.local_efficiency(graph)

#                     # Hubness (using HITS algorithm)
#                     hubs, authorities = nx.hits(graph)
#                     # Formula: HITS hub score: importance of a node as a hub, based on linking to authorities

#                     for chan in chan_list_eeg:

#                         try:
#                             _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'chan' : [chan], 'degree' : [degree[chan]], 'betweenness' : [betweenness[chan]], 
#                                     'closeness' : [closeness[chan]], 'hubs' : [hubs[chan]],
#                                     'clustering_coeff' : [clustering_coeff[chan]], 'local_efficiency' : [local_efficiency]})

#                             df_graph_metrics = pd.concat((df_graph_metrics, _df))
#                         except:
#                             pass

#             os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

#             for metric in ['degree', 'betweenness', 'closeness', 'hubs', 'clustering_coeff', 'local_efficiency']:

#                 g = sns.catplot(
#                     data=df_graph_metrics, kind="bar",
#                     x="chan", y=metric, hue="cond",
#                     alpha=.6, height=6)
#                 # plt.show()

#                 if stretch:
#                     if fc_metric == 'MI':
#                         plt.savefig(f"stretch_{fc_metric}_graph_{metric}.png")
#                     else:
#                         plt.savefig(f"stretch_{fc_metric}_{band}_graph_{metric}.png")
#                 else:
#                     if fc_metric == 'MI':
#                         plt.savefig(f"nostretch_{fc_metric}_graph_{metric}.png")
#                     else:
#                         plt.savefig(f"nostretch_{fc_metric}_{band}_graph_{metric}.png")

#                 plt.close('all')












################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    plot_allsujet_FC_time_stretch()
    plot_allsujet_FC_mat_stretch()
    plot_allsujet_FC_graph_stretch()




