

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False






########################################
######## MUTUAL INFORMATION ########
########################################


#stretch = True
def compute_stats_MI_allsujet(stretch):

    #### verify computation
    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS_stretch.nc')):
            print(f'ALREADY DONE STATS MI STRETCH')
            return

    else:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_allsujet_STATS.nc')):
            print(f'ALREADY DONE STATS MI')
            return

    #### load
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    if stretch:
        MI_allsujet = xr.open_dataarray('MI_allsujet_stretch.nc')
    else:
        MI_allsujet = xr.open_dataarray('MI_allsujet.nc')
    
    if stretch:
        time_vec = MI_allsujet['phase'].values
        time_vec_stats = time_vec
    else:
        time_vec = MI_allsujet['time'].values
        time_vec_stats = time_vec <= 0

    pairs_to_compute = MI_allsujet['pair'].values

    clusters = np.zeros((pairs_to_compute.size, time_vec.size))

    #pair_i, pair = 1, pairs_to_compute[1]
    for pair_i, pair in enumerate(pairs_to_compute):

        print_advancement(pair_i, len(pairs_to_compute))

        data_baseline = MI_allsujet.loc[:, pair, 'VS', :].values
        data_cond = MI_allsujet.loc[:, pair, 'CHARGE', :].values

        _cluster = get_permutation_cluster_1d(data_baseline[:,time_vec_stats], data_cond[:,time_vec_stats], n_surr_fc)

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
        xr_MI_stats.to_netcdf(f'MI_allsujet_STATS_stretch.nc')
    else:
        xr_MI_stats.to_netcdf(f'MI_allsujet_STATS.nc')


    



#stretch = False
def compute_stats_wpli_ispc_allsujet(stretch):

    #fc_metric = 'ISPC'
    for fc_metric in ['ISPC', 'WPLI']:

        #### verify computation
        if stretch:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS_stretch.nc')):
                print(f'stretch:{stretch} {fc_metric} ALREADY DONE')
                continue

        else:

            if os.path.exists(os.path.join(path_precompute, 'FC', fc_metric, f'{fc_metric}_allsujet_STATS.nc')):
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

        pairs_to_compute = fc_allsujet['pairs'].values

        clusters = np.zeros((len(freq_band_fc_list), pairs_to_compute.size, time_vec.size))

        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            print(band)

            #pair_i, pair = 1, pairs_to_compute[1]
            for pair_i, pair in enumerate(pairs_to_compute):

                print_advancement(pair_i, len(pairs_to_compute))

                data_baseline = fc_allsujet.loc[:, band, 'VS', pair, :].values
                data_cond = fc_allsujet.loc[:, band, 'CHARGE', pair, :].values

                _cluster = get_permutation_cluster_1d(data_baseline[:,time_vec_stats], data_cond[:,time_vec_stats], n_surr_fc)

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
        MI_stats_dict = {'band' : freq_band_fc_list, 'pair' : pairs_to_compute, 'time' : time_vec}

        xr_MI_stats = xr.DataArray(data=clusters, dims=MI_stats_dict.keys(), coords=MI_stats_dict.values())
        
        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

        if stretch:
            xr_MI_stats.to_netcdf(f'{fc_metric}_allsujet_STATS_stretch.nc')
        else:
            xr_MI_stats.to_netcdf(f'{fc_metric}_allsujet_STATS.nc')


    









################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    #stretch = False
    for stretch in [True, False]:

        #compute_stats_MI_allsujet()
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_MI_allsujet', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()

        #compute_stats_wpli_ispc_allsujet()
        execute_function_in_slurm_bash('n08_precompute_FC_STATS', 'compute_stats_wpli_ispc_allsujet', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()
        




