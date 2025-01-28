

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



def compute_stats_MI_allsujet():

    #stretch = False
    for stretch in [True, False]:

        #### verify computation
        if stretch:

            if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'allsujet_MI_STATS_stretch.nc')):
                print(f'ALREADY DONE STATS MI STRETCH')
                continue

        else:

            if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'allsujet_MI_STATS.nc')):
                print(f'ALREADY DONE STATS MI')
                continue

        #### load
        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs.nc')
        n_surr = 500
        time_vec = MI_allsujet['time'].values
        
        if stretch:
            time_vec_stats = time_vec
        else:
            time_vec_stats = time_vec <= 0

        pairs_to_compute = MI_allsujet['pair'].values

        clusters = np.zeros((pairs_to_compute.size, time_vec.size))

        #pair_i, pair = 1, pairs_to_compute[1]
        for pair_i, pair in enumerate(pairs_to_compute):

            data_baseline = MI_allsujet.loc[:, pair, 'VS', :].values
            data_cond = MI_allsujet.loc[:, pair, 'CHARGE', :].values

            _cluster = get_permutation_cluster_1d(data_baseline[:,time_vec_stats], data_cond[:,time_vec_stats], n_surr)

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
            xr_MI_stats.to_netcdf(f'allsujet_MI_STATS_stretch.nc')
        else:
            xr_MI_stats.to_netcdf(f'allsujet_MI_STATS.nc')


        










################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    #get_MI_allsujet()
    execute_function_in_slurm_bash('n06_precompute_FC', 'get_MI_allsujet', [], n_core=15, mem='15G')
    




