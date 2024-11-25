
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import gc
import xarray as xr
import seaborn as sns
import cv2
from matplotlib.animation import FuncAnimation

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n04bis_res_allsujet_ERP import *

debug = False



########################################
######## STATS FUNCTIONS ########
########################################

#tf = tf_plot
def get_tf_stats(tf, pixel_based_distrib):

    #### thresh data
    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] < pixel_based_distrib[wavelet_i, 0], tf_thresh[wavelet_i, :] > pixel_based_distrib[wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf.shape[-1])

        plt.pcolormesh(time, frex, tf, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh




################################
######## MOVIE TOPOPLOT ########
################################


def get_movie_ERP(stretch=False):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats = get_cluster_stats_manual_prem(stretch=True)
        time_vec = xr_data['phase'].values
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats = get_cluster_stats_manual_prem(stretch=False)
        time_vec = xr_data['time'].values

    if debug:
        #### for the first 100ms
        time_window = 0.1 #in s
        time_window_i = 0

        _start = int(time_window_i * time_window * srate)
        _stop = int(_start + time_window * srate)

        cluster_chunk = np.zeros((chan_list_eeg_short.size, int(time_window * srate)))
        data_chunk = np.zeros((chan_list_eeg_short.size, int(time_window * srate)))

        #chan_i, chan = 0, chan_list_eeg[0]
        for chan_i, chan in enumerate(chan_list_eeg_short):

            cluster_chunk[chan_i,:] = cluster_stats[chan_i, _start:_stop].values
            data_baseline = xr_data.loc[:, 'VS', chan, : ].mean('sujet').values[_start:_stop]
            data_cond = xr_data.loc[:, 'CHARGE', chan, : ].mean('sujet').values[_start:_stop]
            data_chunk[chan_i,:] = data_cond - data_baseline

        cluster_mask = cluster_chunk.sum(axis=1) > 0
        data_chunk = data_chunk.mean(axis=1)

        fig, ax = plt.subplots()

        mne.viz.plot_topomap(data=data_chunk, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                        mask=cluster_mask, mask_params=mask_params, cmap='seismic', extrapolate='local')

        plt.show()
    
    ####chunk data
    def get_visu_data(time_window):

        time_chunk_points = int(time_window * srate)
        start_window = np.arange(0, time_vec.size, time_chunk_points)
        n_times = np.arange(start_window.size)
        
        cluster_mask_wins = np.zeros((chan_list_eeg_short.size, start_window.size))
        data_chunk_wins = np.zeros((chan_list_eeg_short.size, start_window.size))

        for win_i, win_start in enumerate(start_window):

            if win_start == start_window[-1]:
                continue
            
            win_stop = start_window[win_i+1]

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg_short):

                _cluster_chunk = cluster_stats[chan_i, win_start:win_stop].values
                cluster_mask_wins[chan_i, win_i] = _cluster_chunk.sum() > 0

                _data_baseline = xr_data.loc[:, 'VS', chan, : ].mean('sujet').values[win_start:win_stop]
                _data_cond = xr_data.loc[:, 'CHARGE', chan, : ].mean('sujet').values[win_start:win_stop]
                data_chunk_wins[chan_i,win_i] = (_data_cond - _data_baseline).mean()

        return n_times, start_window, data_chunk_wins, cluster_mask_wins

    time_window = 0.1#in s
    n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data(time_window)
    vmin, vmax = data_chunk_wins.min(), data_chunk_wins.max()

    if debug:

        chan_i, chan = np.where(chan_list_eeg_short == 'Fz')[0][0], 'Fz'
        plt.plot(time_vec[start_window], data_chunk_wins[chan_i,:])
        plt.show()

    # Create topoplot frames
    fig, ax = plt.subplots()
    cbar = True

    def update(frame):
        global cbar
        ax.clear()
        ax.set_title(np.round(time_vec[start_window[frame]], 5))
        data_chunk = data_chunk_wins[:, frame]
        cluster_mask = cluster_mask_wins[:, frame]
        im, cbar = mne.viz.plot_topomap(data=data_chunk, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                        mask=cluster_mask, mask_params=mask_params, vlim=(vmin, vmax), cmap='seismic', extrapolate='local')
        return [im, cbar]

    # Animation
    ani = FuncAnimation(fig, update, frames=n_times, interval=1500)  # Adjust interval as needed
    plt.show()

    os.chdir(path_results)
    ani.save("topomap_animation.mp4", writer="ffmpeg")  # Requires FFmpeg installed




    ######## SUBJECT WISE ########

    if stretch:
        cluster_stats = get_cluster_stats_manual_prem_one_cond_stretch()
    else:
        cluster_stats = get_cluster_stats_manual_prem_one_cond()

    if debug:

        plt.plot(cluster_stats.loc[:, 'CHARGE', 'C3', :].values.sum(axis=0), label='charge')
        plt.plot(cluster_stats.loc[:, 'VS', 'C3', :].values.sum(axis=0), label='VS')
        plt.legend()
        plt.show()

        for chan in chan_list_eeg_short:
            plt.plot(cluster_stats.loc[:, 'CHARGE', chan, :].values.sum(axis=0), label=chan)
        plt.legend()
        plt.show()    

    for cond_i, cond in enumerate(cond_list):

        ####chunk data
        def get_visu_data(time_window):

            time_chunk_points = int(time_window * srate)
            start_window = np.arange(0, time_vec.size, time_chunk_points)
            n_times = np.arange(start_window.size)
            
            cluster_mask_wins = np.zeros((chan_list_eeg_short.size, start_window.size))
            data_chunk_wins = np.zeros((chan_list_eeg_short.size, start_window.size))

            for win_i, win_start in enumerate(start_window):

                if win_start == start_window[-1]:
                    continue
                
                win_stop = start_window[win_i+1]

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg_short):

                    _cluster_chunk = cluster_stats[:, cond_i, chan_i, win_start:win_stop].values
                    cluster_mask_wins[chan_i, win_i] = _cluster_chunk.sum(axis=0).mean()

                    _data_cond = xr_data.loc[:, cond, chan, : ].mean('sujet').values[win_start:win_stop]
                    data_chunk_wins[chan_i,win_i] = _data_cond.mean()

            return n_times, start_window, data_chunk_wins, cluster_mask_wins
        
        time_window = 0.1#in s
        n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data(time_window)
        vmin, vmax = cluster_mask_wins.min(), cluster_mask_wins.max()

        if debug:

            chan_i, chan = np.where(chan_list_eeg_short == 'Fz')[0][0], 'Fz'
            plt.plot(time_vec[start_window], data_chunk_wins[chan_i,:])
            plt.show()

        # Create topoplot frames
        fig, ax = plt.subplots()
        cbar = True

        def update(frame):
            global cbar
            ax.clear()
            ax.set_title(np.round(time_vec[start_window[frame]], 5))
            data_chunk = data_chunk_wins[:, frame]
            cluster_mask = cluster_mask_wins[:, frame]
            im, cbar = mne.viz.plot_topomap(data=data_chunk, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                            vlim=(vmin, vmax), cmap='Reds', extrapolate='local')
            return [im, cbar]

        # Animation
        ani = FuncAnimation(fig, update, frames=n_times, interval=1500)  # Adjust interval as needed
        plt.show()














################################
######## PSEUDO NETWORK ########
################################


def compute_topoplot_IE_network(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        data_stats_cluster_intra = {}

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_stats_cluster_intra[odor] = {}

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                print(subgroup_type, odor, cond)

                data_stats_cluster_intra[odor][cond] = np.zeros((len(chan_list_eeg), time_vec))

                if subgroup_type == 'allsujet':
                    data_cond = xr_data.loc[:, cond, odor, :, :].values
                    
                elif subgroup_type == 'rep':
                    data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

                elif subgroup_type == 'no_rep':
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_stats_cluster_intra[odor][cond][chan_i,:] = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

        #### scale
        cluster_size = np.array([])

        for phase_i, phase in enumerate(['inspi', 'expi']):

            for odor_i, odor in enumerate(['o', '+', '-']):

                for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_intra[odor][cond][chan_i,int(time_vec/2):]*1).sum() / (time_vec/2)*100, 3) )
                        if phase == 'expi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_intra[odor][cond][chan_i,:int(time_vec/2)]*1).sum() / (time_vec/2)*100, 3) )

        vlim = np.percentile(cluster_size, 99)

        #### plot
        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for odor_i, odor in enumerate(['o', '+', '-']):

                    print('intra', phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            perm_vec_phase = data_stats_cluster_intra[odor][cond][chan_i,int(time_vec/2):]
                        if phase == 'expi':
                            perm_vec_phase = data_stats_cluster_intra[odor][cond][chan_i,:int(time_vec/2)]

                        if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                            if phase == 'inspi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)
                            if phase == 'expi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)

                            mask_signi[chan_i] = True

                    ax = axs[odor_i, phase_i]

                    ax.set_title(f"{odor} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{cond} {subgroup_type} INTRA {np.round(vlim,2)}')

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))
            fig.savefig(f"{subgroup_type}_intra_{cond}.jpeg")

            plt.close('all')
            
            # plt.show()

        ######## INTER ########
        #### load stats
        data_stats_cluster_inter = {}

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_stats_cluster_inter[cond] = {}

            data_baseline = xr_data.loc[:, cond, 'o', :, :].values

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, cond, 'o', :, :].values
                
            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                print(odor, cond)

                data_stats_cluster_inter[cond][odor] = np.zeros((len(chan_list_eeg), time_vec))

                if subgroup_type == 'allsujet':
                    data_cond = xr_data.loc[:, cond, odor, :, :].values
                    
                elif subgroup_type == 'rep':
                    data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

                elif subgroup_type == 'no_rep':
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_stats_cluster_inter[cond][odor][chan_i,:] = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

        #### scale
        cluster_size = np.array([])

        for phase_i, phase in enumerate(['inspi', 'expi']):

            for odor_i, odor in enumerate(['+', '-']):

                for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_inter[cond][odor][chan_i,int(time_vec/2):]*1).sum()/(time_vec/2)*100, 3) )
                        if phase == 'expi':
                            cluster_size = np.append( cluster_size, np.round((data_stats_cluster_inter[cond][odor][chan_i,:int(time_vec/2)]*1).sum()/(time_vec/2)*100, 3) )

        vlim = np.percentile(cluster_size, 99)

        #### plot
        for odor_i, odor in enumerate(['+', '-']):    

            fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15,15))

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

                    print('intra', phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        if phase == 'inspi':
                            perm_vec_phase = data_stats_cluster_inter[cond][odor][chan_i,int(time_vec/2):]
                        if phase == 'expi':
                            perm_vec_phase = data_stats_cluster_inter[cond][odor][chan_i,:int(time_vec/2)]

                        if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                            if phase == 'inspi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)
                            if phase == 'expi':
                                data_topoplot[chan_i] = np.round(perm_vec_phase.sum()/(time_vec/2)*100, 3)

                            mask_signi[chan_i] = True

                    ax = axs[cond_i, phase_i]

                    ax.set_title(f"{cond} {phase}")

                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask=mask_signi, mask_params=mask_params, vlim=(0, vlim), cmap='seismic')

            plt.tight_layout()

            plt.suptitle(f'{odor} {subgroup_type} INTER {np.round(vlim,2)}')

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))
            fig.savefig(f"{subgroup_type}_inter_{odor}.jpeg")

            plt.close('all')
            
            # plt.show()







def compute_topoplot_IE_network_SUM(stretch=False):

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats = get_cluster_stats_manual_prem(stretch=True)
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats_intra = get_cluster_stats_manual_prem()

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]
    
    #### scale
    cluster_size = np.array([])

    for phase_i, phase in enumerate(['inspi', 'expi']):

        for odor_i, odor in enumerate(['o', '+', '-']):

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                #chan_i, chan = 6, chan_list_eeg[6]
                for chan_i, chan in enumerate(chan_list_eeg):

                    if phase == 'inspi':
                        perm_vec_phase = np.concatenate([np.zeros(int(time_vec/2)), cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[int(time_vec/2):]], axis=0)
                    if phase == 'expi':
                        perm_vec_phase = np.concatenate([cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[:int(time_vec/2)], np.zeros(int(time_vec/2))], axis=0) 

                    if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                        if subgroup_type == 'allsujet':
                            data_baseline = np.mean(xr_data.loc[:, 'FR_CV_1', odor, chan, :].values, axis=0)

                        elif subgroup_type == 'rep':
                            data_baseline = np.mean(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                        elif subgroup_type == 'no_rep':
                            data_baseline = np.mean(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                        if subgroup_type == 'allsujet':
                            data_cond = np.mean(xr_data.loc[:, cond, odor, chan, :].values, axis=0)
                            
                        elif subgroup_type == 'rep':
                            data_cond = np.mean(xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values, axis=0)

                        elif subgroup_type == 'no_rep':
                            data_cond = np.mean(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values, axis=0)

                        data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                        cluster_size = np.append(cluster_size, data_diff_sel.sum())

    vlim = np.percentile(cluster_size, 99)

    cluster_size_allphase = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            #chan_i, chan = 6, chan_list_eeg[6]
            for chan_i, chan in enumerate(chan_list_eeg):

                perm_vec_phase = cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :]

                if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                    if subgroup_type == 'allsujet':
                        data_baseline = np.mean(xr_data.loc[:, 'FR_CV_1', odor, chan, :].values, axis=0)

                    elif subgroup_type == 'rep':
                        data_baseline = np.mean(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                    elif subgroup_type == 'no_rep':
                        data_baseline = np.mean(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                    if subgroup_type == 'allsujet':
                        data_cond = np.mean(xr_data.loc[:, cond, odor, chan, :].values, axis=0)
                        
                    elif subgroup_type == 'rep':
                        data_cond = np.mean(xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values, axis=0)

                    elif subgroup_type == 'no_rep':
                        data_cond = np.mean(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values, axis=0)

                    data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                    cluster_size_allphase = np.append(cluster_size_allphase, data_diff_sel.sum())

    vlim_allphase = np.percentile(cluster_size_allphase, 99)

    #### get allchan response
    allchan_response_intra = {'region' : [], 'cond' : [], 'phase' : [], 'odor' : [], 'sum' : []}

    for region in chan_list_lobes:

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

            for phase_i, phase in enumerate(['inspi', 'expi']):

                for odor_i, odor in enumerate(['o', '+', '-']):

                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_lobes[region]):

                        if phase == 'inspi':
                            perm_vec_phase = np.concatenate([np.zeros(int(time_vec/2)), cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[int(time_vec/2):]], axis=0)
                        if phase == 'expi':
                            perm_vec_phase = np.concatenate([cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[:int(time_vec/2)], np.zeros(int(time_vec/2))], axis=0) 

                        if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                            if subgroup_type == 'allsujet':
                                data_baseline = np.mean(xr_data.loc[:, 'FR_CV_1', odor, chan, :].values, axis=0)

                            elif subgroup_type == 'rep':
                                data_baseline = np.mean(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                            elif subgroup_type == 'no_rep':
                                data_baseline = np.mean(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                            if subgroup_type == 'allsujet':
                                data_cond = np.mean(xr_data.loc[:, cond, odor, chan, :].values, axis=0)
                                
                            elif subgroup_type == 'rep':
                                data_cond = np.mean(xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values, axis=0)

                            elif subgroup_type == 'no_rep':
                                data_cond = np.mean(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values, axis=0)

                            data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                            data_topoplot[chan_i] = data_diff_sel.sum()

                    allchan_response_intra['cond'].append(cond)
                    allchan_response_intra['phase'].append(phase)
                    allchan_response_intra['odor'].append(odor)
                    allchan_response_intra['sum'].append(data_topoplot.sum()/len(chan_list_lobes[region]))
                    allchan_response_intra['region'].append(region)
    
    df_allchan_response_intra = pd.DataFrame(allchan_response_intra)

    #### plot inspi expi
    print('plot intra')
    
    #cond_i, cond = 1, 'CO2'
    for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

        for phase_i, phase in enumerate(['inspi', 'expi']):

            for odor_i, odor in enumerate(['o', '+', '-']):

                print('intra', phase, odor, cond)

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                data_topoplot = np.zeros((len(chan_list_eeg)))

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    if phase == 'inspi':
                        perm_vec_phase = np.concatenate([np.zeros(int(time_vec/2)), cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[int(time_vec/2):]], axis=0)
                    if phase == 'expi':
                        perm_vec_phase = np.concatenate([cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values[:int(time_vec/2)], np.zeros(int(time_vec/2))], axis=0) 

                    if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                        if subgroup_type == 'allsujet':
                            data_baseline = np.mean(xr_data.loc[:, 'FR_CV_1', odor, chan, :].values, axis=0)

                        elif subgroup_type == 'rep':
                            data_baseline = np.mean(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                        elif subgroup_type == 'no_rep':
                            data_baseline = np.mean(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                        if subgroup_type == 'allsujet':
                            data_cond = np.mean(xr_data.loc[:, cond, odor, chan, :].values, axis=0)
                            
                        elif subgroup_type == 'rep':
                            data_cond = np.mean(xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values, axis=0)

                        elif subgroup_type == 'no_rep':
                            data_cond = np.mean(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values, axis=0)

                        data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                        data_topoplot[chan_i] = data_diff_sel.sum()

                        mask_signi[chan_i] = True

                ax = axs[odor_i, phase_i]

                ax.set_title(f"{odor} {phase}")

                mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

        plt.tight_layout()

        plt.suptitle(f'{cond} {subgroup_type} INTRA {np.round(vlim,2)}')

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

        if stretch:
            fig.savefig(f"stretch_SUM_{subgroup_type}_intra_{cond}.jpeg")
        else:
            fig.savefig(f"SUM_{subgroup_type}_intra_{cond}.jpeg")

        plt.close('all')
        
        # plt.show()

    #### plot allphase    
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

    for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

        for odor_i, odor in enumerate(['o', '+', '-']):

            print('intra allphase', odor, cond)

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
            data_topoplot = np.zeros((len(chan_list_eeg)))

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                perm_vec_phase = cluster_stats_intra.loc[subgroup_type, chan, odor, cond, :].values
                
                if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                    if subgroup_type == 'allsujet':
                        data_baseline = np.mean(xr_data.loc[:, 'FR_CV_1', odor, chan, :].values, axis=0)

                    elif subgroup_type == 'rep':
                        data_baseline = np.mean(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                    elif subgroup_type == 'no_rep':
                        data_baseline = np.mean(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, chan, :].values, axis=0)

                    if subgroup_type == 'allsujet':
                        data_cond = np.mean(xr_data.loc[:, cond, odor, chan, :].values, axis=0)
                        
                    elif subgroup_type == 'rep':
                        data_cond = np.mean(xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values, axis=0)

                    elif subgroup_type == 'no_rep':
                        data_cond = np.mean(xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values, axis=0)

                    data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                    data_topoplot[chan_i] = data_diff_sel.sum()

                    mask_signi[chan_i] = True

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{odor} {cond}")

            mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim_allphase, vlim_allphase), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'{subgroup_type} INTRA {np.round(vlim_allphase,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network'))

    if stretch:
        fig.savefig(f"stretch_SUM_ALLPHASE_{subgroup_type}_intra.jpeg")
    else:
        fig.savefig(f"SUM_ALLPHASE_{subgroup_type}_intra.jpeg")

    plt.close('all')
    
    # plt.show()

    #### plot allchan response
    for region in chan_list_lobes:

        for cond in ['MECA', 'CO2', 'FR_CV_2']:

            sns.barplot(data=df_allchan_response_intra.query(f"cond == '{cond}' and region == '{region}'"), x='phase', y='sum', hue='odor', hue_order=["o", "+", "-"], order=['expi', 'inspi'])

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
            plt.title(f'{subgroup_type} intra {cond} {region}')
            
            if stretch:
                plt.savefig(f"stretch_ALLCHAN_{subgroup_type}_{cond}_{region}_intra.jpeg")
            else:
                plt.savefig(f"ALLCHAN_{subgroup_type}_{cond}_{region}_intra.jpeg")

            plt.close('all')

    sns.barplot(data=df_allchan_response_intra.query(f"cond == 'CO2' and odor == 'o'"), x='region', y='sum', hue='phase', hue_order=['expi', 'inspi'])

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_IE_network', 'allchan_sum'))
    plt.title(f'{subgroup_type} intra CO2')
    
    if stretch:
        plt.savefig(f"stretch_ALLCHAN_ALLREGION_{subgroup_type}_{cond}_intra.jpeg")
    else:
        plt.savefig(f"ALLCHAN_ALLREGION_{subgroup_type}_{cond}_intra.jpeg")

    plt.close('all')

    #### values for CO2
    df_allchan_response_intra.query(f"cond == 'CO2' and region == 'all'")

        








################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### MAIN WORKFLOW

    get_movie_ERP(stretch=False)

    compute_topoplot_IE_network_SUM(stretch=False)
    compute_topoplot_IE_network_SUM(stretch=True)





