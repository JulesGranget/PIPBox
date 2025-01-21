
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
from n04_precompute_ERP import *

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
        cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=True)
        time_vec = xr_data['phase'].values
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=False)
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

                _cluster_chunk = cluster_stats.loc[chan,:].values[win_start:win_stop]
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
    ani = FuncAnimation(fig, update, frames=n_times, interval=1500)
    plt.show()

    os.chdir(os.path.join(path_results, 'ERP', 'topoplot'))
    ani.save("ERP_topomap_animation_allsujet.gif", writer="pillow")  




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





def compute_topoplot_ERP_network_SUM():

    xr_data, xr_data_sem = compute_ERP()
    cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=False)

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]

    #### scale
    cluster_size = np.array([])

    #chan_i, chan = 0, chan_list_eeg_short[0]
    for chan_i, chan in enumerate(chan_list_eeg_short):

        perm_vec_phase = cluster_stats.loc[chan, :].values

        if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
            data_baseline = np.mean(xr_data.loc[:, 'VS', chan, :].values, axis=0)
            data_cond = np.mean(xr_data.loc[:, 'CHARGE', chan, :].values, axis=0)

            data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
            cluster_size = np.append(cluster_size, data_diff_sel.sum())

    vlim = np.percentile(cluster_size, 99)

    #### plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))

    mask_signi = np.zeros(len(chan_list_eeg_short)).astype('bool')
    data_topoplot = np.zeros((len(chan_list_eeg_short)))

    #chan_i, chan = 0, chan_list_eeg_short[0]
    for chan_i, chan in enumerate(chan_list_eeg_short):

        perm_vec_phase = cluster_stats.loc[chan, :].values

        if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

            data_baseline = np.mean(xr_data.loc[:, 'VS', chan, :].values, axis=0)
            data_cond = np.mean(xr_data.loc[:, 'CHARGE', chan, :].values, axis=0)

            data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
            data_topoplot[chan_i] = data_diff_sel.sum()

            mask_signi[chan_i] = True

    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                    mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic', extrapolate='local')

    plt.title(f'allsujet n:{xr_data.shape[0]} lim:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'ERP', 'topoplot'))

    fig.savefig(f"SUM_allsujet.jpeg")

    plt.close('all')
    
    # plt.show()

    
    


def timing_ERP_SUM_plot():   

    xr_data, xr_data_sem = compute_ERP()
    cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=False)

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data['time'].data
    time_vec_PPI = time_vec[time_vec <= 0]
    start_time, stop_time = -3, 0

    ######## PLOT RESPONSE FOR ALL GROUPS ########

    #### load stats
    timing_data = np.zeros((len(chan_list_eeg_short), len(['value', 'time'])))

    data_baseline = xr_data.loc[:, 'VS', chan_list_eeg_short, :].values
    data_cond = xr_data.loc[:, 'CHARGE', chan_list_eeg_short, :].values
        
    #chan_i, chan = 5, chan_list_eeg_short[5]
    for chan_i, chan in enumerate(chan_list_eeg_short):

        data_baseline_chan = data_baseline[:, chan_i, :]
        data_cond_chan = data_cond[:, chan_i, :] 

        mask_signi = cluster_stats.loc[chan,:].values

        if mask_signi.sum() == 0:
            continue     

        else:
            mask_signi[0] = False
            mask_signi[-1] = False   

        if np.diff(mask_signi).sum() > 2: 

            start_stop_chunk = np.where(np.diff(mask_signi))[0]

            max_chunk_signi = []
            max_chunk_time = []
            for start_i in np.arange(0, start_stop_chunk.size, 2):

                _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                max_chunk_signi.append(data_cond_chan.mean(axis=0)[_argmax])

            max_rep = max_chunk_signi[np.argmax(np.abs(max_chunk_signi))]
            time_max_rep = time_vec[max_chunk_time[np.where(max_chunk_signi == max_rep)[0][0]]]

        else:

            start_stop_chunk = np.where(np.diff(mask_signi))[0]

            _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[0]:start_stop_chunk[1]])
            max_rep = data_cond_chan.mean(axis=0)[start_stop_chunk[0] + _argmax]
            time_max_rep = time_vec[start_stop_chunk[0] +_argmax]

        timing_data[chan_i, 0] = max_rep
        timing_data[chan_i, 1] = time_max_rep

    xr_coords = {'chan' : chan_list_eeg_short, 'type' : ['value', 'time']}
    xr_timing = xr.DataArray(data=timing_data, coords=xr_coords)     

    ### plot
    min, max = xr_timing.loc[:,'value'].data.min(), xr_timing.loc[:,'value'].data.max()
    vlim = np.max([np.abs(min), np.abs(max)])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,8))

    angles_vec = np.linspace(0, 2*np.pi, num=time_vec_PPI.size)
    mask_sel = (xr_timing.loc[:, 'time'] != 0).data
    time_responses_filter = xr_timing.loc[:, 'time'].data[mask_sel]
    ampl_responses_filter = xr_timing.loc[:, 'value'].data[mask_sel]

    time_responses_i = [np.where(time_vec_PPI == time_val)[0][0] for time_val in time_responses_filter]
    angle_responses = angles_vec[time_responses_i]

    _phase_mean = np.angle(np.mean(np.exp(1j*angle_responses)))
    
    mean_vec = np.round(time_vec_PPI[[angle_i for angle_i, angle_val in enumerate(angles_vec) if np.mod(_phase_mean, 2 * np.pi) < angle_val][0]], 2)

    ax.scatter(angle_responses, ampl_responses_filter, s=10)

    ax.plot(angles_vec, np.zeros((time_vec_PPI.size)), color='k')

    ax.plot([_phase_mean, _phase_mean], [0, np.mean(ampl_responses_filter)])

    ax.set_xticks(np.linspace(0, 2 * np.pi, num=4, endpoint=False))
    ax.set_xticklabels(np.round(np.linspace(start_time,stop_time, num=4, endpoint=False), 2))
    ax.set_yticks(np.round(np.linspace(-vlim,vlim, num=3, endpoint=True), 2))
    ax.set_rlim([-vlim,vlim])

    plt.title(f'ERP response time allsujet time:{mean_vec} vlim:{np.round(vlim,2)}')

    # plt.show()

    os.chdir(os.path.join(path_results, 'ERP', 'time'))
    plt.savefig(f"ERP_time_allsujet.png")







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






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    compute_topoplot_ERP_network_SUM()
    timing_ERP_SUM_plot()





    get_movie_ERP(stretch=False)

    compute_topoplot_IE_network_SUM(stretch=False)
    compute_topoplot_IE_network_SUM(stretch=True)





