

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

    print(f'compute {fc_metric} stretch:{stretch}')

    #### load
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    cond_sel = ['VS', 'CHARGE']
    MI_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

    for sujet_i, sujet in enumerate(sujet_list_FC):

        _MI_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
        MI_allsujet[sujet_i] = _MI_sujet.values

    MI_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

    xr_MI_allsujet = xr.DataArray(data=MI_allsujet, dims=MI_allsujet_dict.keys(), coords=MI_allsujet_dict.values())

    phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
    phase_shift = int(stretch_point_FC/4) 
    # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
    phase_vec = {'whole' : np.arange(stretch_point_FC), 'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                 'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 

    shifted_xr_MI_allsujet = xr_MI_allsujet.roll(time=-phase_shift, roll_coords=False)

    shifted_xr_MI_allsujet = xr_MI_allsujet.median('ntrials')

    # pvals_wk = np.zeros((len(phase_list), pairs_to_compute.size))
    pvals_perm = np.zeros((len(phase_list), len(pairs_to_compute)))

    #phase_i, phase = 0, phase_list[0]
    for phase_i, phase in enumerate(phase_list):

        print(phase)

        #pair_i, pair = 0, pairs_to_compute[0]
        for pair_i, pair in enumerate(pairs_to_compute):

            # print_advancement(pair_i, len(pairs_to_compute))

            data_baseline = np.median(shifted_xr_MI_allsujet.loc[:, 'VS', pair, phase_vec[phase]].values, axis=1)
            data_cond = np.median(shifted_xr_MI_allsujet.loc[:, 'CHARGE', pair, phase_vec[phase]].values, axis=1)

            # stat, pvals_wk[phase_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
            pvals_perm[phase_i, pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc,
                                                                  mode_grouped='median', mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5])

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
    fc_stats_dict = {'phase' : phase_list, 'pair' : pairs_to_compute}

    xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

    xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state_stretch.nc')

        
        



# def compute_stats_MI_allsujet_state_nostretch():

#     fc_metric = 'MI'

#     if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_state.nc')):
#         print(f'ALREADY DONE STATS {fc_metric}')
#         return

#     print(f'compute {fc_metric} stretch:{stretch}')

#     #### load
#     os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
#     fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')

#     time_vec = fc_allsujet['time'].values
#     time_vec_stats = time_vec <= 0

#     pairs_to_compute = fc_allsujet['pair'].values

#     # pvals_wk = np.zeros((pairs_to_compute.size))
#     pvals_perm = np.zeros((pairs_to_compute.size))

#     #pair_i, pair = 0, pairs_to_compute[0]
#     for pair_i, pair in enumerate(pairs_to_compute):

#         # print_advancement(pair_i, len(pairs_to_compute))

#         data_baseline = np.median(fc_allsujet.loc[:, pair, 'VS', time_vec_stats].values, axis=1)
#         data_cond = np.median(fc_allsujet.loc[:, pair, 'CHARGE', time_vec_stats].values, axis=1)

#         if debug:

#             plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
#             plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
#             plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
#             plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
#             plt.legend()
#             plt.show()

#         # stat, pvals_wk[pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
#         pvals_perm[pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc)


#     # Apply Benjamini-Hochberg correction
#     # reject, pvals_adjusted, _, _ = multipletests(pvals_wk, alpha=0.05, method='fdr_bh')

#     if debug:

#         plt.plot(pvals_perm, label='perm')
#         # plt.plot(reject, label='Benjamini-Hochberg')
#         plt.legend()
#         plt.show()

#     #### export
#     fc_stats_dict = {'pair' : pairs_to_compute}

#     xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
    
#     os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

#     xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state.nc')

    
    

#stretch = True
def compute_stats_ispc_wpli_allsujet_state(stretch):

    #fc_metric = 'WPLI'
    for fc_metric in ['WPLI', 'ISPC']:

        #### verify computation
        if stretch:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_state_stretch.nc')):
                print(f'ALREADY DONE STATS {fc_metric} STRETCH')
                continue

        else:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_state.nc')):
                print(f'ALREADY DONE STATS {fc_metric}')
                continue

        print(f'compute {fc_metric} stretch:{stretch}')

        #### load
        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        if stretch:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
        else:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')

        time_vec = fc_allsujet['time'].values
        
        if stretch:
            time_vec_stats = time_vec
        else:
            time_vec_stats = time_vec <= 0

        pairs_to_compute = fc_allsujet['pair'].values

        if stretch:

            phase_list = ['whole', 'I', 'T_IE', 'E', 'T_EI']
            phase_shift = 125 
            # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
            phase_vec = {'whole' : np.arange(stretch_point_ERP), 'I' : np.arange(250), 'T_IE' : np.arange(250)+250, 'E' : np.arange(250)+500, 'T_EI' : np.arange(250)+750} 

            # pvals_wk = np.zeros((len(phase_list), len(freq_band_fc_list), pairs_to_compute.size))
            pvals_perm = np.zeros((len(phase_list), len(freq_band_fc_list), pairs_to_compute.size))

            shifted_fc_allsujet = fc_allsujet.roll(time=-phase_shift, roll_coords=False)

            for phase_i, phase in enumerate(phase_list):

                for band_i, band in enumerate(freq_band_fc_list):

                    print(phase, band)

                    #pair_i, pair = 0, pairs_to_compute[0]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        # print_advancement(pair_i, len(pairs_to_compute))

                        data_baseline = np.median(shifted_fc_allsujet.loc[:, band, 'VS', pair, phase_vec[phase]].values, axis=1)
                        data_cond = np.median(shifted_fc_allsujet.loc[:, band, 'CHARGE', pair, phase_vec[phase]].values, axis=1)

                        # stat, pvals_wk[band_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                        pvals_perm[phase_i, band_i, pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc, 
                                                                                      mode_grouped='median', mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5])

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
            fc_stats_dict = {'phase' : phase_list, 'band' : freq_band_fc_list, 'pair' : pairs_to_compute}

            xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
            
            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state_stretch.nc')

        else:

            # pvals_wk = np.zeros((len(freq_band_fc_list), pairs_to_compute.size))
            pvals_perm = np.zeros((len(freq_band_fc_list), pairs_to_compute.size))

            for band_i, band in enumerate(freq_band_fc_list):

                print(band)

                #pair_i, pair = 0, pairs_to_compute[0]
                for pair_i, pair in enumerate(pairs_to_compute):

                    # print_advancement(pair_i, len(pairs_to_compute))

                    data_baseline = np.median(fc_allsujet.loc[:, band, 'VS', pair, time_vec_stats].values, axis=1)
                    data_cond = np.median(fc_allsujet.loc[:, band, 'CHARGE', pair, time_vec_stats].values, axis=1)

                    if debug:

                        plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
                        plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
                        plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
                        plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                        plt.legend()
                        plt.show()

                    # stat, pvals_wk[band_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                    pvals_perm[band_i, pair_i] = get_permutation_2groups(data_baseline, data_cond, n_surr_fc, 
                                                                         mode_grouped='median', mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5])

            # Apply Benjamini-Hochberg correction
            # for band_i in range(len(freq_band_fc_list)):
            #     pvals_wk[band_i,:], pvals_adjusted, _, _ = multipletests(pvals_wk[band_i,:], alpha=0.05, method='fdr_bh')

            if debug:

                plt.plot(pvals_perm, label='perm')
                # plt.plot(pvals_wk, label='Benjamini-Hochberg')
                plt.legend()
                plt.show()

            #### export
            fc_stats_dict = {'band' : freq_band_fc_list, 'pair' : pairs_to_compute}

            xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
            
            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state.nc')


    








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

    cond_sel = ['VS', 'CHARGE']
    MI_allsujet = np.zeros((len(sujet_list_FC), len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

    for sujet_i, sujet in enumerate(sujet_list_FC):

        _MI_sujet = xr.open_dataarray(f'MI_stretch_{sujet}.nc')
        MI_allsujet[sujet_i] = _MI_sujet.values

    MI_allsujet_dict = {'sujet' : sujet_list_FC, 'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

    xr_MI_allsujet = xr.DataArray(data=MI_allsujet, dims=MI_allsujet_dict.keys(), coords=MI_allsujet_dict.values())

    time_vec = xr_MI_allsujet['time'].values

    clusters_allsujet = np.zeros((len(pairs_to_compute), time_vec.size))

    #pair_i, pair = 5, pairs_to_compute[5]
    for pair_i, pair in enumerate(pairs_to_compute):

        print_advancement(pair_i, len(pairs_to_compute))

        data_baseline = np.median(xr_MI_allsujet.loc[:, 'VS', pair, :].values, axis=1)
        data_cond = np.median(xr_MI_allsujet.loc[:, 'CHARGE', pair, :].values, axis=1)

        data_baseline_rscore = (data_baseline - np.median(data_baseline, axis=1).reshape(-1,1)) * 0.6745 / scipy.stats.median_abs_deviation(data_baseline, axis=1).reshape(-1,1)
        data_cond_rscore = (data_cond - np.median(data_cond, axis=1).reshape(-1,1)) * 0.6745 / scipy.stats.median_abs_deviation(data_cond, axis=1).reshape(-1,1)

        if debug:

            for i in range(len(sujet_list_FC)):
                plt.plot(data_baseline_rscore[i,:], alpha=0.2)
            plt.plot(np.median(data_baseline_rscore, axis=0), color='k')
            plt.show()

            for i in range(len(sujet_list_FC)):
                plt.plot(data_cond_rscore[i,:], alpha=0.2)
            plt.plot(np.median(data_cond_rscore, axis=0), color='k')
            plt.show()

            plt.hist(data_baseline_rscore.reshape(-1), bins=50, alpha=0.5, label='VS')
            plt.hist(data_cond_rscore.reshape(-1), bins=50, alpha=0.5, label='CHARGE')
            plt.legend()
            plt.show()

        _cluster_pair = get_permutation_cluster_1d(data_baseline_rscore, data_cond_rscore, n_surr_fc,
                                                   mode_grouped='median', mode_generate_surr='percentile_time', 
                                                   mode_select_thresh='percentile_time', size_thresh_alpha=0.01)

        if debug:

            mode = 'median'

            if mode == 'mean':
                data_baseline_grouped = np.mean(data_baseline, axis=0)
                data_cond_grouped = np.mean(data_cond, axis=0)
            elif mode == 'median':
                data_baseline_grouped = np.median(data_baseline_rscore, axis=0)
                data_cond_grouped = np.median(data_cond_rscore, axis=0)
            
            time = np.arange(data_baseline_rscore.shape[-1])
            mad_baseline = scipy.stats.median_abs_deviation(data_baseline_rscore, axis=0)
            mad_cond = scipy.stats.median_abs_deviation(data_cond_rscore, axis=0)

            min = np.array([data_baseline_grouped-mad_baseline, data_cond_grouped-mad_cond]).min()
            max = np.array([data_baseline_grouped+mad_baseline, data_cond_grouped+mad_cond]).max()

            plt.plot(time, data_baseline_grouped, label='baseline', color='c')
            plt.fill_between(time, data_baseline_grouped-mad_baseline, data_baseline_grouped+mad_baseline, color='c', alpha=0.5)
            plt.plot(time, data_cond_grouped, label='cond', color='g')
            plt.fill_between(time, data_cond_grouped-mad_cond, data_cond_grouped+mad_cond, color='g', alpha=0.5)
            plt.fill_between(time, min, max, where=_cluster_pair, color='r', alpha=0.5)
            plt.title(f'{pair} {pair_i}')
            plt.legend()
            plt.show()

        clusters_allsujet[pair_i, :] = _cluster_pair 

    #### export
    MI_stats_dict = {'pair' : pairs_to_compute, 'time' : time_vec}

    xr_MI_stats = xr.DataArray(data=clusters_allsujet, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_time_stretch.nc')


    



# def compute_stats_MI_allsujet_time_nostretch():

#     #### verify computation
#     if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time.nc')):
#         print(f'ALREADY DONE STATS MI')
#         return

#     #### load
#     os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
#     MI_allsujet = xr.open_dataarray('MI_allsujet.nc')

#     time_vec = MI_allsujet['time'].values
#     time_vec_stats = time_vec <= 0

#     pairs_to_compute = MI_allsujet['pair'].values

#     clusters = np.zeros((pairs_to_compute.size, time_vec.size))

#     #pair_i, pair = 5, pairs_to_compute[5]
#     for pair_i, pair in enumerate(pairs_to_compute):

#         print_advancement(pair_i, len(pairs_to_compute))

#         data_baseline = MI_allsujet.loc[:, pair, 'VS', time_vec_stats].values
#         data_cond = MI_allsujet.loc[:, pair, 'CHARGE', time_vec_stats].values

#         if debug:

#             for i in range(len(sujet_list)):
#                 plt.plot(data_baseline[i,:], alpha=0.2)
#             plt.plot(data_baseline.mean(axis=0), color='k')
#             plt.show()

#             for i in range(len(sujet_list)):
#                 plt.plot(data_cond[i,:], alpha=0.2)
#             plt.plot(data_cond.mean(axis=0), color='k')
#             plt.show()

#             plt.hist(data_baseline.reshape(-1), bins=50, alpha=0.5, label='VS')
#             plt.hist(data_cond.reshape(-1), bins=50, alpha=0.5, label='CHARGE')
#             plt.legend()
#             plt.show()

#         _cluster = get_permutation_cluster_1d(data_baseline, data_cond, n_surr_fc)

#         if debug:

#             mode = 'mean'

#             if mode == 'mean':
#                 data_baseline_grouped = np.mean(data_baseline, axis=0)
#                 data_cond_grouped = np.mean(data_cond, axis=0)
#             elif mode == 'median':
#                 data_baseline_grouped = np.median(data_baseline, axis=0)
#                 data_cond_grouped = np.median(data_cond, axis=0)
            
#             time = np.arange(data_baseline.shape[-1])
#             sem_baseline = data_baseline.std(axis=0)/np.sqrt(data_baseline.shape[0])
#             sem_cond = data_cond.std(axis=0)/np.sqrt(data_cond.shape[0])

#             plt.plot(time, data_baseline_grouped, label='baseline', color='c')
#             plt.fill_between(time, data_baseline_grouped-sem_baseline, data_baseline_grouped+sem_baseline, color='c', alpha=0.5)
#             plt.plot(time, data_cond_grouped, label='cond', color='g')
#             plt.fill_between(time, data_cond_grouped-sem_cond, data_cond_grouped+sem_cond, color='g', alpha=0.5)
#             plt.fill_between(time, data_baseline_grouped.min(), data_cond_grouped.max(), where=_cluster, color='r', alpha=0.5)
#             plt.title(f'{pair} {pair_i}')
#             plt.legend()
#             plt.show()

#         _cluster = np.concatenate((_cluster, np.zeros((time_vec.size - time_vec[time_vec_stats].size)).astype('bool')))

#         if debug:

#             min, max = np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).min(), np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).max()
#             fig, ax = plt.subplots()
#             ax.plot(time_vec, data_baseline.mean(axis=0), label='VS')
#             ax.plot(time_vec, data_cond.mean(axis=0), label='CHARGE')
#             ax.fill_between(time_vec, min, max, where=_cluster.astype('int'), alpha=0.3, color='r')
#             plt.show()

#         clusters[pair_i, :] = _cluster 

#     #### export
#     MI_stats_dict = {'pair' : pairs_to_compute, 'time' : time_vec}

#     xr_MI_stats = xr.DataArray(data=clusters, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
    
#     os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
#     xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_time.nc')


    



#stretch = True
def compute_stats_wpli_ispc_allsujet_time(stretch):

    #fc_metric = 'ISPC'
    for fc_metric in ['ISPC', 'WPLI']:

        #### verify computation
        if stretch:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_time_stretch.nc')):
                print(f'stretch:{stretch} {fc_metric} ALREADY DONE')
                continue

        else:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_time.nc')):
                print(f'stretch:{stretch} {fc_metric} ALREADY DONE')
                continue
            
        print(f'COMPUTE stretch:{stretch} {fc_metric}')

        #### load
        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        if stretch:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
        else:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
        
        time_vec = fc_allsujet['time'].values

        if stretch:    
            time_vec_stats = time_vec
        else:
            time_vec_stats = time_vec <= 0

        pairs_to_compute = fc_allsujet['pair'].values

        clusters = np.zeros((len(freq_band_fc_list), pairs_to_compute.size, time_vec.size))

        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            print(band)

            #pair_i, pair = 1, pairs_to_compute[1]
            for pair_i, pair in enumerate(pairs_to_compute):

                print_advancement(pair_i, len(pairs_to_compute))

                data_baseline = fc_allsujet.loc[:, band, 'VS', pair, time_vec_stats].values
                data_cond = fc_allsujet.loc[:, band, 'CHARGE', pair, time_vec_stats].values

                _cluster = get_permutation_cluster_1d(data_baseline, data_cond, n_surr_fc)

                if not stretch:
                    _cluster = np.concatenate((_cluster, np.zeros((time_vec.size - time_vec[time_vec_stats].size)).astype('bool')))

                if debug:

                    min, max = np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).min(), np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).max()
                    fig, ax = plt.subplots()
                    ax.plot(time_vec, data_baseline.mean(axis=0), label='VS')
                    ax.plot(time_vec, data_cond.mean(axis=0), label='CHARGE')
                    ax.fill_between(time_vec, min, max, where=_cluster.astype('int'), alpha=0.3, color='r')
                    plt.show()

                clusters[band_i, pair_i, :] = _cluster 

        #### export
        fc_stats_dict = {'band' : freq_band_fc_list, 'pair' : pairs_to_compute, 'time' : time_vec}

        xr_fc_stats = xr.DataArray(data=clusters, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
        
        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

        if stretch:
            xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_time_stretch.nc')
        else:
            xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_time.nc')


    









################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    compute_stats_MI_allsujet_state_stretch()
    compute_stats_ispc_wpli_allsujet_state()
    compute_stats_MI_allsujet_time_stretch()
    compute_stats_ispc_wpli_allsujet_state()

    #stretch = False
    for stretch in [True, False]:

        #compute_stats_MI_allsujet_time(stretch)
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_MI_allsujet_time', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()

        #compute_stats_wpli_ispc_allsujet_time(stretch)
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_wpli_ispc_allsujet_time', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()
        




