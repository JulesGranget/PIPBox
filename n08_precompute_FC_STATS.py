

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




#stretch = True
def compute_stats_MI_allsujet_state(stretch):

    #fc_metric = 'MI'
    for fc_metric in ['MI']:

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

            phase_list = ['whole', 'inspi', 'expi']
            phase_vec = {'whole' : time_vec, 'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

            # pvals_wk = np.zeros((len(phase_list), pairs_to_compute.size))
            pvals_perm = np.zeros((len(phase_list), pairs_to_compute.size))

            for phase_i, phase in enumerate(phase_list):

                print(phase)

                #pair_i, pair = 0, pairs_to_compute[0]
                for pair_i, pair in enumerate(pairs_to_compute):

                    # print_advancement(pair_i, len(pairs_to_compute))

                    data_baseline = np.median(fc_allsujet.loc[:, pair, 'VS', phase_vec[phase]].values, axis=1)
                    data_cond = np.median(fc_allsujet.loc[:, pair, 'CHARGE', phase_vec[phase]].values, axis=1)

                    if debug:

                        plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
                        plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
                        plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
                        plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                        plt.legend()
                        plt.show()

                    # stat, pvals_wk[phase_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                    pvals_perm[phase_i, pair_i] = get_permutation_wilcoxon_2groups(data_baseline, data_cond, n_surr_fc)

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

        else:

            # pvals_wk = np.zeros((pairs_to_compute.size))
            pvals_perm = np.zeros((pairs_to_compute.size))

            #pair_i, pair = 0, pairs_to_compute[0]
            for pair_i, pair in enumerate(pairs_to_compute):

                # print_advancement(pair_i, len(pairs_to_compute))

                data_baseline = np.median(fc_allsujet.loc[:, pair, 'VS', time_vec_stats].values, axis=1)
                data_cond = np.median(fc_allsujet.loc[:, pair, 'CHARGE', time_vec_stats].values, axis=1)

                if debug:

                    plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
                    plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
                    plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
                    plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                    plt.legend()
                    plt.show()

                # stat, pvals_wk[pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                pvals_perm[pair_i] = get_permutation_wilcoxon_2groups(data_baseline, data_cond, n_surr_fc)


            # Apply Benjamini-Hochberg correction
            # reject, pvals_adjusted, _, _ = multipletests(pvals_wk, alpha=0.05, method='fdr_bh')

            if debug:

                plt.plot(pvals_perm, label='perm')
                # plt.plot(reject, label='Benjamini-Hochberg')
                plt.legend()
                plt.show()

            #### export
            fc_stats_dict = {'pair' : pairs_to_compute}

            xr_fc_stats = xr.DataArray(data=pvals_perm, dims=fc_stats_dict.keys(), coords=fc_stats_dict.values())
            
            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            xr_fc_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_state.nc')
    

#stretch = True
def compute_stats_ispc_wpli_allsujet_state(stretch):

    #fc_metric = 'ISPC'
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

            phase_list = ['whole', 'inspi', 'expi']
            phase_vec = {'whole' : time_vec, 'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

            # pvals_wk = np.zeros((len(phase_list), len(freq_band_fc_list), pairs_to_compute.size))
            pvals_perm = np.zeros((len(phase_list), len(freq_band_fc_list), pairs_to_compute.size))

            for phase_i, phase in enumerate(phase_list):

                for band_i, band in enumerate(freq_band_fc_list):

                    print(phase, band)

                    #pair_i, pair = 0, pairs_to_compute[0]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        # print_advancement(pair_i, len(pairs_to_compute))

                        data_baseline = np.median(fc_allsujet.loc[:, band, 'VS', pair, phase_vec[phase]].values, axis=1)
                        data_cond = np.median(fc_allsujet.loc[:, band, 'CHARGE', pair, phase_vec[phase]].values, axis=1)

                        if debug:

                            plt.hist(data_baseline, bins=50, alpha=0.5, label='VS')
                            plt.hist(data_cond, bins=50, alpha=0.5, label='CHARGE')
                            plt.vlines([np.median(data_baseline)], ymin=0, ymax=10, color='r')
                            plt.vlines([np.median(data_cond)], ymin=0, ymax=10, color='r')
                            plt.legend()
                            plt.show()

                        # stat, pvals_wk[band_i, pair_i] = scipy.stats.wilcoxon(data_baseline, data_cond)
                        pvals_perm[phase_i, band_i, pair_i] = get_permutation_wilcoxon_2groups(data_baseline, data_cond, n_surr_fc)

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
                    pvals_perm[band_i, pair_i] = get_permutation_wilcoxon_2groups(data_baseline, data_cond, n_surr_fc)

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




#stretch = True
def compute_stats_MI_allsujet_time(stretch):

    #### verify computation
    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time_stretch.nc')):
            print(f'ALREADY DONE STATS MI STRETCH')
            return

    else:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time.nc')):
            print(f'ALREADY DONE STATS MI')
            return

    #### load
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    if stretch:
        MI_allsujet = xr.open_dataarray('MI_allsujet_stretch.nc')
    else:
        MI_allsujet = xr.open_dataarray('MI_allsujet.nc')


    #### verify computation
    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time_stretch.nc')):
            print(f'ALREADY DONE STATS MI STRETCH')
            return

    else:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_time.nc')):
            print(f'ALREADY DONE STATS MI')
            return

    #### load
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    if stretch:
        MI_allsujet = xr.open_dataarray('MI_allsujet_stretch.nc')
    else:
        MI_allsujet = xr.open_dataarray('MI_allsujet.nc')

    time_vec = MI_allsujet['time'].values
    
    if stretch:
        time_vec_stats = time_vec
    else:
        time_vec_stats = time_vec <= 0

    pairs_to_compute = MI_allsujet['pair'].values

    clusters = np.zeros((pairs_to_compute.size, time_vec.size))

    #pair_i, pair = 5, pairs_to_compute[5]
    for pair_i, pair in enumerate(pairs_to_compute):

        print_advancement(pair_i, len(pairs_to_compute))

        data_baseline = MI_allsujet.loc[:, pair, 'VS', time_vec_stats].values
        data_cond = MI_allsujet.loc[:, pair, 'CHARGE', time_vec_stats].values

        if debug:

            for i in range(len(sujet_list)):
                plt.plot(data_baseline[i,:], alpha=0.2)
            plt.plot(data_baseline.mean(axis=0), color='k')
            plt.show()

            for i in range(len(sujet_list)):
                plt.plot(data_cond[i,:], alpha=0.2)
            plt.plot(data_cond.mean(axis=0), color='k')
            plt.show()

            plt.hist(data_baseline.reshape(-1), bins=50, alpha=0.5, label='VS')
            plt.hist(data_cond.reshape(-1), bins=50, alpha=0.5, label='CHARGE')
            plt.legend()
            plt.show()

        _cluster = get_permutation_cluster_1d(data_baseline, data_cond, n_surr_fc)

        if debug:

            mode = 'mean'

            if mode == 'mean':
                data_baseline_grouped = np.mean(data_baseline, axis=0)
                data_cond_grouped = np.mean(data_cond, axis=0)
            elif mode == 'median':
                data_baseline_grouped = np.median(data_baseline, axis=0)
                data_cond_grouped = np.median(data_cond, axis=0)
            
            time = np.arange(data_baseline.shape[-1])
            sem_baseline = data_baseline.std(axis=0)/np.sqrt(data_baseline.shape[0])
            sem_cond = data_cond.std(axis=0)/np.sqrt(data_cond.shape[0])

            plt.plot(time, data_baseline_grouped, label='baseline', color='c')
            plt.fill_between(time, data_baseline_grouped-sem_baseline, data_baseline_grouped+sem_baseline, color='c', alpha=0.5)
            plt.plot(time, data_cond_grouped, label='cond', color='g')
            plt.fill_between(time, data_cond_grouped-sem_cond, data_cond_grouped+sem_cond, color='g', alpha=0.5)
            plt.fill_between(time, data_baseline_grouped.min(), data_cond_grouped.max(), where=_cluster, color='r', alpha=0.5)
            plt.title(f'{pair} {pair_i}')
            plt.legend()
            plt.show()

        if not stretch:
            _cluster = np.concatenate((_cluster, np.zeros((time_vec.size - time_vec[time_vec_stats].size)).astype('bool')))

        if debug:

            min, max = np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).min(), np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).max()
            fig, ax = plt.subplots()
            ax.plot(time_vec, data_baseline.mean(axis=0), label='VS')
            ax.plot(time_vec, data_cond.mean(axis=0), label='CHARGE')
            ax.fill_between(time_vec, min, max, where=_cluster.astype('int'), alpha=0.3, color='r')
            plt.show()

        clusters[pair_i, :] = _cluster 

    #### export
    MI_stats_dict = {'pair' : pairs_to_compute, 'time' : time_vec}

    xr_MI_stats = xr.DataArray(data=clusters, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

    if stretch:
        xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_time_stretch.nc')
    else:
        xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_time.nc')


    



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

    #stretch = False
    for stretch in [True, False]:

        compute_stats_MI_allsujet_state(stretch)
        compute_stats_ispc_wpli_allsujet_state(stretch)

    #stretch = False
    for stretch in [True, False]:

        #compute_stats_MI_allsujet_time(stretch)
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_MI_allsujet_time', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()

        #compute_stats_wpli_ispc_allsujet_time(stretch)
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_wpli_ispc_allsujet_time', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()
        




