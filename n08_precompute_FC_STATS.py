

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import scipy.stats
import xarray as xr
from statsmodels.stats.multitest import multipletests

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False




########################################
######## FC STATE ANALYSIS ########
########################################



def compute_stats_MI_allsujet_state_stretch():

    fc_metric = 'MI'

    #### verify computation
    if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_state_stretch.nc')):
        print(f'ALREADY DONE STATS {fc_metric} STRETCH')
        return

    print(f'compute {fc_metric}')

    #### load
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')    

    print('load data')    

    cond_sel = ['VS', 'CHARGE']
    MI_allsujet_phase_norm = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), stretch_point_FC))
    MI_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), stretch_point_FC))

    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

    for sujet_i, sujet in enumerate(sujet_list_FC):

        print_advancement(sujet_i, len(sujet_list))

        _MI_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
        MI_allsujet[sujet_i] = _MI_sujet.median('ntrials').values
        _MI_sujet_rscore = xr.open_dataarray(f'MI_stretch_{sujet}_rscore.nc')
        MI_allsujet_phase_norm[sujet_i] = np.median(_MI_sujet_rscore, axis=2)

    MI_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}
    xr_MI_allsujet_rscore = xr.DataArray(data=MI_allsujet_phase_norm, dims=MI_allsujet_dict.keys(), coords=MI_allsujet_dict.values())

    xr_MI_allsujet = xr.DataArray(data=MI_allsujet, dims=MI_allsujet_dict.keys(), coords=MI_allsujet_dict.values())

    if debug:

        pair = pairs_to_compute[2]

        for pair in pairs_to_compute:

            diff_plot = xr_MI_allsujet.loc[:, 'CHARGE', pair, :].values - xr_MI_allsujet.loc[:, 'VS', pair, :].values 
            for sujet_i in range(diff_plot.shape[0]):
                plt.plot(diff_plot[sujet_i], alpha=0.2)
            plt.plot(np.median(diff_plot, axis=0), color='r')            
            plt.show()

            diff_plot = xr_MI_allsujet.loc[:, 'CHARGE', pair, :].values - xr_MI_allsujet.loc[:, 'VS', pair, :].values 
            diff_plot_rscore = diff_plot.copy()
            for sujet_i in range(diff_plot.shape[0]):
                diff_plot_rscore[sujet_i] = (diff_plot_rscore[sujet_i] - np.median(diff_plot_rscore[sujet_i])) * 0.6745 / scipy.stats.median_abs_deviation(diff_plot_rscore[sujet_i])
            
            for sujet_i in range(diff_plot.shape[0]):
                plt.plot(diff_plot_rscore[sujet_i], alpha=0.2)
            plt.plot(np.median(diff_plot_rscore, axis=0), color='r')            
            plt.show()

    phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
    phase_shift = int(stretch_point_FC/4) 
    # 0-60, 60-120, 120-180, 180-240
    phase_vec = {'whole' : np.arange(stretch_point_FC), 'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                 'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 

    shifted_xr_MI_allsujet = xr_MI_allsujet.roll(time=-phase_shift, roll_coords=False)
    shifted_xr_MI_allsujet_rscore = xr_MI_allsujet_rscore.roll(time=-phase_shift, roll_coords=False)

    print('compute stats')

    # pvals_wk = np.zeros((len(phase_list), pairs_to_compute.size))
    pvals_perm = np.zeros((2, len(phase_list), len(pairs_to_compute)))
    pvals_perm_mne = np.zeros((2, len(phase_list), len(pairs_to_compute)))

    for data_type_i, data_type in enumerate(['raw', 'rscore']):

        #phase_i, phase = 1, phase_list[1]
        for phase_i, phase in enumerate(phase_list):

            print(phase)

            #pair_i, pair = 0, pairs_to_compute[0]
            for pair_i, pair in enumerate(pairs_to_compute):

                # print_advancement(pair_i, len(pairs_to_compute))

                if data_type == 'raw':
                    data_baseline = np.median(shifted_xr_MI_allsujet.loc[:, 'VS', pair, phase_vec[phase]].values, axis=-1)
                    data_cond = np.median(shifted_xr_MI_allsujet.loc[:, 'CHARGE', pair, phase_vec[phase]].values, axis=-1)
                else:
                    data_baseline = np.median(shifted_xr_MI_allsujet_rscore.loc[:, 'VS', pair, phase_vec[phase]].values, axis=-1)
                    data_cond = np.median(shifted_xr_MI_allsujet_rscore.loc[:, 'CHARGE', pair, phase_vec[phase]].values, axis=-1)

                if debug:
                
                    plt.hist(data_baseline, alpha=0.5)
                    plt.hist(data_cond, alpha=0.5)
                    plt.show()

                # stat, pvals_wk[phase_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                pvals_perm[data_type_i, phase_i, pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc, stat_design=stat_design, mode_grouped=mode_grouped, 
                                                                    mode_generate_surr=mode_generate_surr_2g, percentile_thresh=percentile_thresh)
                
                T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_cond - data_baseline, n_permutations=n_surr_fc, threshold=None,
                                                                    tail=0, out_type="mask", verbose=False)
                if cluster_p_values.size == 0:
                    pvals_perm_mne[data_type_i, phase_i, pair_i] = False
                else:
                    pvals_perm_mne[data_type_i, phase_i, pair_i] = True

                if debug:

                    plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
                    plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
                    plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
                    plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                    plt.legend()
                    plt.show()

    # Apply Benjamini-Hochberg correction
    # reject, pvals_adjusted, _, _ = multipletests(pvals_wk, alpha=0.05, method='fdr_bh')

    if debug:

        plt.plot(pvals_perm, label='perm')
        # plt.plot(reject, label='Benjamini-Hochberg')
        plt.legend()
        plt.show()

    #### export
    fc_stats_dict = {'data_type' : ['raw', 'rscore'], 'phase' : phase_list, 'pair' : pairs_to_compute}

    xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
    xr_fc_stats_mne = xr.DataArray(data=pvals_perm_mne, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

    xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
    xr_fc_stats_mne.to_netcdf(f'{fc_metric}_mne_allsujet_STATS_state_stretch.nc')

        
        


def compute_stats_ispc_wpli_allsujet_state_stretch():

    #fc_metric = 'ISPC'
    for fc_metric in ['WPLI', 'ISPC']:

        #### verify computation
        if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_state_stretch.nc')):
            print(f'ALREADY DONE STATS')
            continue

        print(f'load data {fc_metric}')

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')        

        cond_sel = ['VS', 'CHARGE']

        fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), stretch_point_FC))
        fc_allsujet_phase_norm = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), stretch_point_FC))

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

        for sujet_i, sujet in enumerate(sujet_list_FC):

            print_advancement(sujet_i, len(sujet_list))

            _fc_allsujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch.nc')
            fc_allsujet[sujet_i] = _fc_allsujet.median('cycle').values
            _fc_allsujet_rscore = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')            
            fc_allsujet_phase_norm[sujet_i] = np.median(_fc_allsujet_rscore, axis=3)

        fc_allsujet_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'cond' : cond_sel, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

        xr_fc_allsujet_rscore = xr.DataArray(data=fc_allsujet_phase_norm, dims=fc_allsujet_allsujet_dict.keys(), coords=fc_allsujet_allsujet_dict.values())
        xr_fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_allsujet_dict.keys(), coords=fc_allsujet_allsujet_dict.values())

        if debug:

            pair = pairs_to_compute[2]

            for pair in pairs_to_compute:

                diff_plot = xr_fc_allsujet.loc[:, 'CHARGE', pair, :].values - xr_fc_allsujet.loc[:, 'VS', pair, :].values 
                for sujet_i in range(diff_plot.shape[0]):
                    plt.plot(diff_plot[sujet_i], alpha=0.2)
                plt.plot(np.median(diff_plot, axis=0), color='r')            
                plt.show()

                diff_plot = xr_fc_allsujet.loc[:, 'CHARGE', pair, :].values - xr_fc_allsujet.loc[:, 'VS', pair, :].values 
                diff_plot_rscore = diff_plot.copy()
                for sujet_i in range(diff_plot.shape[0]):
                    diff_plot_rscore[sujet_i] = (diff_plot_rscore[sujet_i] - np.median(diff_plot_rscore[sujet_i])) * 0.6745 / scipy.stats.median_abs_deviation(diff_plot_rscore[sujet_i])
                
                for sujet_i in range(diff_plot.shape[0]):
                    plt.plot(diff_plot_rscore[sujet_i], alpha=0.2)
                plt.plot(np.median(diff_plot_rscore, axis=0), color='r')            
                plt.show()


        phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
        phase_shift = int(stretch_point_FC/4) 
        # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
        phase_vec = {'whole' : np.arange(stretch_point_FC), 'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                    'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 

        xr_fc_allsujet = xr_fc_allsujet.roll(time=-phase_shift, roll_coords=False)
        xr_fc_allsujet_rscore = xr_fc_allsujet_rscore.roll(time=-phase_shift, roll_coords=False)

        print('compute stats')

        # pvals_wk = np.zeros((len(phase_list), len(freq_band_fc_list), pairs_to_compute.size))
        pvals_perm = np.zeros((2, len(phase_list), len(freq_band_fc_list), len(pairs_to_compute)))
        pvals_perm_mne = np.zeros((2, len(phase_list), len(freq_band_fc_list), len(pairs_to_compute)))

        for data_type_i, data_type in enumerate(['raw', 'rscore']):

            for phase_i, phase in enumerate(phase_list):

                for band_i, band in enumerate(freq_band_fc_list):

                    print(phase, band)

                    #pair_i, pair = 0, pairs_to_compute[0]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        # print_advancement(pair_i, len(pairs_to_compute))

                        if data_type == 'raw':
                            data_baseline = np.median(xr_fc_allsujet.loc[:, pair, 'VS', band, phase_vec[phase]].values, axis=-1)
                            data_cond = np.median(xr_fc_allsujet.loc[:, pair, 'CHARGE', band, phase_vec[phase]].values, axis=-1)
                        else:
                            data_baseline = np.median(xr_fc_allsujet_rscore.loc[:, pair, 'VS', band, phase_vec[phase]].values, axis=-1)
                            data_cond = np.median(xr_fc_allsujet_rscore.loc[:, pair, 'CHARGE', band, phase_vec[phase]].values, axis=-1)

                        if debug:

                            plt.plot(data_baseline)
                            plt.plot(data_cond)
                            plt.show()

                        # stat, pvals_wk[band_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                        pvals_perm[data_type_i, phase_i, band_i, pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc, stat_design=stat_design, mode_grouped=mode_grouped, 
                                                                    mode_generate_surr=mode_generate_surr_2g, percentile_thresh=percentile_thresh)
                        
                        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_cond - data_baseline, n_permutations=n_surr_fc, threshold=None,
                                                                    tail=0, out_type="mask", verbose=False)
                        if cluster_p_values.size == 0:
                            pvals_perm_mne[data_type_i, phase_i, band_i, pair_i] = False
                        else:
                            pvals_perm_mne[data_type_i, phase_i, band_i, pair_i] = True

                        if debug:

                            plt.hist(data_baseline, bins=50, alpha=0.5, label='VS', color='b')
                            plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE', color='r')
                            plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='b')
                            plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                            plt.title(f"signi:{pvals_perm[phase_i, band_i, pair_i]}")
                            plt.legend()
                            plt.show()


        # Apply Benjamini-Hochberg correction
        # for band_i in range(len(freq_band_fc_list)):
        #     pvals_wk[band_i,:], pvals_adjusted, _, _ = multipletests(pvals_wk[band_i,:], alpha=0.05, method='fdr_bh')

        if debug:

            plt.plot(pvals_perm, label='perm')
            # plt.plot(pvals_wk, label='Benjamini-Hochberg')
            plt.legend()
            plt.show()

        #### export
        fc_stats_dict = {'data_type' : ['raw', 'rscore'], 'phase' : phase_list, 'band' : freq_band_fc_list, 'pair' : pairs_to_compute}

        xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
        xr_fc_stats_mne = xr.DataArray(data=pvals_perm_mne, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
        
        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

        xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
        xr_fc_stats_mne.to_netcdf(f'{fc_metric}_mne_allsujet_STATS_state_stretch.nc')

        






########################################
######## FC TIME ANALYSIS ########
########################################


def compute_stats_MI_allsujet_time_stretch():

    #### verify computation
    if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time_stretch.nc')):
        print(f'ALREADY DONE STATS MI STRETCH')
        return

    #### load
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    print('load data')    

    for data_type in ['raw', 'rscore']:

        cond_sel = ['VS', 'CHARGE']
        MI_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), stretch_point_FC))

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

        for sujet_i, sujet in enumerate(sujet_list_FC):

            print_advancement(sujet_i, len(sujet_list_FC))

            if data_type == 'raw':
                _MI_sujet_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
            else:
                _MI_sujet_sujet = xr.open_dataarray(f'MI_stretch_{sujet}_rscore.nc')   

            MI_allsujet[sujet_i] = np.median(_MI_sujet_sujet, axis=2)

        MI_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}
        xr_MI_allsujet = xr.DataArray(data=MI_allsujet, dims=MI_allsujet_dict.keys(), coords=MI_allsujet_dict.values())

        time_vec = xr_MI_allsujet['time'].values

        print('compute stats')

        clusters_allsujet = np.zeros((len(pairs_to_compute), time_vec.size))
        clusters_allsujet_mne = np.zeros((len(pairs_to_compute), time_vec.size))

        #pair_i, pair = 5, pairs_to_compute[5]
        for pair_i, pair in enumerate(pairs_to_compute):

            print_advancement(pair_i, len(pairs_to_compute))

            data_baseline = xr_MI_allsujet.loc[:, 'VS', pair, :].values
            data_cond = xr_MI_allsujet.loc[:, 'CHARGE', pair, :].values

            if debug:

                for i in range(len(sujet_list_FC)):
                    plt.plot(data_baseline[i,:], alpha=0.2)
                plt.plot(np.median(data_baseline, axis=0), color='k')
                plt.show()

                for i in range(len(sujet_list_FC)):
                    plt.plot(data_cond[i,:], alpha=0.2)
                plt.plot(np.median(data_cond, axis=0), color='k')
                plt.show()

                for i in range(len(sujet_list_FC)):
                    plt.plot(data_cond[i,:] - data_baseline[i,:], alpha=0.2)
                plt.plot(np.median(data_cond - data_baseline, axis=0), color='k')
                plt.show()

                plt.hist(data_baseline.reshape(-1), bins=50, alpha=0.5, label='VS')
                plt.hist(data_cond.reshape(-1), bins=50, alpha=0.5, label='CHARGE')
                plt.legend()
                plt.show()

            _cluster_pair = get_permutation_cluster_1d(data_baseline, data_cond, n_surr_fc, stat_design=stat_design, mode_grouped=mode_grouped, 
                                                mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                                size_thresh_alpha=size_thresh_alpha)
            
            clusters_allsujet[pair_i, :] = _cluster_pair 
        
            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_cond - data_baseline, n_permutations=n_surr_fc, threshold=None,
                                                                        tail=0, out_type="indices", verbose=False)
            
            if cluster_p_values.size != 0 and any(cluster_p_values < 0.05):
                for cluster_i in np.where(cluster_p_values < 0.05)[0]:
                    clusters_allsujet_mne[pair_i, clusters[cluster_i][0]] = 1
            
            if debug:

                data_diff = data_cond - data_baseline
                min, max = data_diff.min(), data_diff.max()
                time = np.arange(data_diff.shape[-1])

                for i in range(len(sujet_list_FC)):
                    plt.plot(data_diff[i,:], alpha=0.2)
                plt.plot(np.median(data_diff, axis=0), color='k')
                plt.fill_between(time, min, max, where=_cluster_pair, color='r', alpha=0.5)
                plt.title(f'{pair} {pair_i}')
                plt.show()

        #### export
        MI_stats_dict = {'pair' : pairs_to_compute, 'time' : time_vec}

        xr_MI_stats = xr.DataArray(data=clusters_allsujet, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
        xr_MI_stats_mne = xr.DataArray(data=clusters_allsujet_mne, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
        
        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        if data_type == 'raw':
            xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_HOMEMADE_time_stretch.nc')
            xr_MI_stats_mne.to_netcdf(f'MI_allsujet_STATS_MNE_time_stretch.nc')
        if data_type == 'rscore':
            xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_HOMEMADE_time_stretch_rscore.nc')
            xr_MI_stats_mne.to_netcdf(f'MI_allsujet_STATS_MNE_time_stretch_rscore.nc')

    


def compute_stats_wpli_ispc_allsujet_time_stretch():

    #fc_metric = 'WPLI'
    for fc_metric in ['ISPC', 'WPLI']:

        #### verify computation
        if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_time_stretch.nc')):
            print(f'stretch:{fc_metric} ALREADY DONE')
            continue
            
        print(f'COMPUTE stretch:{fc_metric}')

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')        

        cond_sel = ['VS', 'CHARGE']

        for data_type in ['raw', 'rscore']:

            fc_allsujet_phase = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), stretch_point_FC))

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            for sujet_i, sujet in enumerate(sujet_list_FC):

                print_advancement(sujet_i, len(sujet_list))

                if data_type == 'raw':
                    _fc_allsujet_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch.nc')
                else:
                    _fc_allsujet_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')

                fc_allsujet_phase[sujet_i] = np.median(_fc_allsujet_sujet, axis=3)

            fc_allsujet_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'cond' : cond_sel, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

            xr_fc_allsujet = xr.DataArray(data=fc_allsujet_phase, dims=fc_allsujet_allsujet_dict.keys(), coords=fc_allsujet_allsujet_dict.values())

            print('compute stats')

            clusters = np.zeros((len(freq_band_fc_list), len(pairs_to_compute), stretch_point_FC))
            clusters_mne = np.zeros((len(freq_band_fc_list), len(pairs_to_compute), stretch_point_FC))

            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                print(band)

                #pair_i, pair = 1, pairs_to_compute[1]
                for pair_i, pair in enumerate(pairs_to_compute):

                    print_advancement(pair_i, len(pairs_to_compute))

                    data_baseline = xr_fc_allsujet.loc[:, pair, 'VS', band, :].values
                    data_cond = xr_fc_allsujet.loc[:, pair, 'CHARGE', band, :].values

                    _cluster = get_permutation_cluster_1d(data_baseline, data_cond, n_surr_fc, stat_design=stat_design, mode_grouped=mode_grouped, 
                                                mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                                size_thresh_alpha=size_thresh_alpha)
                    
                    clusters[band_i, pair_i, :] = _cluster

                    T_obs, _clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_cond - data_baseline, n_permutations=n_surr_fc, threshold=None,
                                                                        tail=0, out_type="indices", verbose=False)
            
                    if cluster_p_values.size != 0 and any(cluster_p_values < 0.05):
                        for cluster_i in np.where(cluster_p_values < 0.05)[0]:
                            clusters_mne[band_i, pair_i, _clusters[cluster_i][0]] = 1

                    if debug:

                        data_diff = data_cond - data_baseline
                        min, max = data_diff.min(), data_diff.max()
                        time = np.arange(data_diff.shape[-1])

                        for i in range(len(sujet_list_FC)):
                            plt.plot(data_diff[i,:], alpha=0.2)
                        plt.plot(np.median(data_diff, axis=0), color='k')
                        plt.fill_between(time, min, max, where=_cluster, color='r', alpha=0.5)
                        plt.title(f'{pair} {pair_i}')
                        plt.show()

            #### export
            fc_stats_dict = {'band' : freq_band_fc_list, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}

            xr_fc_stats = xr.DataArray(data=clusters, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
            xr_fc_stats_mne = xr.DataArray(data=clusters_mne, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
            
            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
            if data_type == 'raw':
                xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_HOMEMADE_time_stretch.nc')
                xr_fc_stats_mne.to_netcdf(f'{fc_metric}_allsujet_STATS_MNE_time_stretch.nc')
            else:
                xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_HOMEMADE_time_stretch_rscore.nc')
                xr_fc_stats_mne.to_netcdf(f'{fc_metric}_allsujet_STATS_MNE_time_stretch_rscore.nc')


    






################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    compute_stats_MI_allsujet_state_stretch()
    compute_stats_ispc_wpli_allsujet_state_stretch()
    compute_stats_MI_allsujet_time_stretch()
    compute_stats_wpli_ispc_allsujet_time_stretch()

        




