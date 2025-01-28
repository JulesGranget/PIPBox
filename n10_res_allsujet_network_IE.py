
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
    # plt.show()

    os.chdir(os.path.join(path_results, 'ERP', 'topoplot'))
    if stretch:
        ani.save("STRETCH_ERP_topomap_animation_allsujet.gif", writer="pillow")  
    else:
        ani.save("ERP_topomap_animation_allsujet.gif", writer="pillow")  













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




def compute_topoplot_ERP_network_SUM_stretch():

    xr_data, xr_data_sem = compute_ERP_stretch()
    cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=True)

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data.shape[-1]

    phase_resp_list = ['inspi', 'expi']
    phase_dict = {'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

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
    for phase_resp in phase_resp_list:

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15))

        mask_signi = np.zeros(len(chan_list_eeg_short)).astype('bool')
        data_topoplot = np.zeros((len(chan_list_eeg_short)))

        #chan_i, chan = 0, chan_list_eeg_short[0]
        for chan_i, chan in enumerate(chan_list_eeg_short):

            perm_vec_phase = cluster_stats.loc[chan, phase_dict[phase_resp]].values

            if perm_vec_phase.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 

                data_baseline = np.mean(xr_data.loc[:, 'VS', chan, phase_dict[phase_resp]].values, axis=0)
                data_cond = np.mean(xr_data.loc[:, 'CHARGE', chan, phase_dict[phase_resp]].values, axis=0)

                data_diff_sel = data_cond[perm_vec_phase.astype('bool')] - data_baseline[perm_vec_phase.astype('bool')]
                data_topoplot[chan_i] = data_diff_sel.sum()

                mask_signi[chan_i] = True

        mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                        mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic', extrapolate='local')

        plt.title(f'allsujet n:{xr_data.shape[0]} lim:{np.round(vlim,2)}')

        os.chdir(os.path.join(path_results, 'ERP', 'topoplot'))

        fig.savefig(f"STRETCH_SUM_{phase_resp}_allsujet.jpeg")

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











################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    compute_topoplot_ERP_network_SUM()
    compute_topoplot_ERP_network_SUM_stretch()
    
    timing_ERP_SUM_plot()

    get_movie_ERP(stretch=False)
    get_movie_ERP(stretch=True)







