
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc
import xarray as xr
import seaborn as sns
import pickle

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test
from mne.stats import spatio_temporal_cluster_1samp_test

debug = False












################################
######## ERP ANALYSIS ########
################################


def compute_ERP():

    if os.path.exists(os.path.join(path_precompute, 'ERP', 'allsujet_ERP_data.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data = xr.open_dataarray('allsujet_ERP_data.nc')
        xr_data_sem = xr.open_dataarray('allsujet_ERP_data_sem.nc')

    else:

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

        xr_dict = {'sujet' : sujet_list, 'cond' : cond_list, 'nchan' : chan_list_eeg, 'time' : time_vec}
        xr_data = xr.DataArray(data=np.zeros((len(sujet_list), len(cond_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_data_sem = xr.DataArray(data=np.zeros((len(sujet_list), len(cond_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        #sujet_i, sujet = 0, sujet_list[0]
        for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):

                #cond = 'VS'
                for cond_i, cond in enumerate(cond_list):

                    #### load
                    data = load_data_sujet(sujet, cond)
                    data = data[:len(chan_list_eeg),:]

                    respfeatures_i = respfeatures[cond]
                    inspi_starts = respfeatures_i['inspi_index'].values

                    #### chunk
                    data_ERP = np.zeros((inspi_starts.shape[0], time_vec.size))

                    #### low pass 45Hz + detrend
                    x = data[nchan_i,:]
                    x = scipy.signal.detrend(x, type='linear')
                    x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)
                    x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                    for start_i, start_time in enumerate(inspi_starts):

                        t_start = int(start_time + t_start_PPI*srate)
                        t_stop = int(start_time + t_stop_PPI*srate)

                        if t_start < 0 or t_stop > x.size:
                            continue

                        x_chunk = x[t_start: t_stop]

                        data_ERP[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()

                    if debug:

                        plt.plot(x)
                        plt.show()

                        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                        for inspi_i, _ in enumerate(inspi_starts):

                            plt.plot(time_vec, data_ERP[inspi_i, :], alpha=0.3)

                        plt.vlines(0, ymax=data_ERP.max(), ymin=data_ERP.min(), color='k')
                        plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                        plt.plot(time_vec, data_ERP.mean(axis=0), color='r')
                        plt.title(f'{cond} : {data_ERP.shape[0]}')
                        plt.show()

                    #### clean
                    if debug:

                        data_stretch_clean = data_ERP[~((data_ERP >= 3) | (data_ERP <= -3)).any(axis=1),:]

                        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                        for inspi_i, _ in enumerate(inspi_starts):

                            plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                        plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                        plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                        plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                        plt.title(f'{cond} : {data_stretch_clean.shape[0]}')
                        plt.show()

                    xr_data.loc[sujet, cond, nchan, :] = data_ERP.mean(axis=0)
                    xr_data_sem.loc[sujet, cond, nchan, :] = data_ERP.std(axis=0) / np.sqrt(data_ERP.shape[0])

        #### save data
        os.chdir(os.path.join(path_precompute, 'ERP'))
        xr_data.to_netcdf('allsujet_ERP_data.nc')
        xr_data_sem.to_netcdf('allsujet_ERP_data_sem.nc')

    return xr_data, xr_data_sem
            





def compute_ERP_stretch():

    if os.path.exists(os.path.join(path_precompute, 'ERP', 'allsujet_ERP_data_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data_stretch = xr.open_dataarray('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch = xr.open_dataarray('allsujet_ERP_data_sem_stretch.nc')

    else:

        xr_dict_stretch = {'sujet' : sujet_list, 'cond' : cond_list, 'nchan' : chan_list_eeg, 'phase' : np.arange(stretch_point_ERP)}
        
        os.chdir(path_memmap)
        data_stretch_ERP = np.memmap(f'data_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(chan_list_eeg), stretch_point_ERP))
        data_sem_stretch_ERP = np.memmap(f'data_sem_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(chan_list_eeg), stretch_point_ERP))

        #sujet_i, sujet = 0, sujet_list[0]
        def get_stretch_data_for_ERP(sujet_i, sujet):

        #sujet_i, sujet = np.where(sujet_list == '12BD')[0][0], '12BD'
        # for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = np.where(chan_list_eeg == 'FC1')[0][0], 'FC1'
            for nchan_i, nchan in enumerate(chan_list_eeg):

                #cond_i, cond = 1, 'CHARGE'
                for cond_i, cond in enumerate(cond_list):

                    data = load_data_sujet(sujet, cond)[nchan_i,:]
                    data = scipy.signal.detrend(data, type='linear')
                    # data = zscore(data)
                    data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond], stretch_point_ERP, data, srate)

                    # data_stretch = zscore_mat(data_stretch)

                    if debug:

                        plt.plot(data, label='raw')
                        plt.plot(scipy.signal.detrend(data, type='linear'), label='detrend')
                        plt.legend()
                        plt.show()

                        plt.hist(data, label='raw', bins=100)
                        plt.hist(scipy.signal.detrend(data, type='linear'), label='detrend', bins=100)
                        plt.legend()
                        plt.show()

                        fig, ax = plt.subplots()

                        for inspi_i in range(data_stretch.shape[0]):

                            ax.plot(np.arange(stretch_point_ERP), data_stretch[inspi_i, :], alpha=0.3)

                        plt.vlines(stretch_point_ERP/2, ymax=data_stretch.max(), ymin=data_stretch.min(), color='k')
                        ax.plot(np.arange(stretch_point_ERP), data_stretch.mean(axis=0), color='r')
                        plt.title(f'{cond} : {data_stretch.shape[0]}')
                        ax.invert_yaxis()
                        plt.show()

                    # data_stretch_load = data_stretch.mean(axis=0)
                    # data_stretch_sem_load = data_stretch.std(axis=0) / np.sqrt(data_stretch.shape[0])

                    data_stretch_ERP[sujet_i, cond_i, nchan_i, :] = data_stretch.mean(axis=0)
                    data_sem_stretch_ERP[sujet_i, cond_i, nchan_i, :] = data_stretch.std(axis=0) / np.sqrt(data_stretch.shape[0])

                    # inverse to have inspi on the right and expi on the left
                    # data_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_load[int(stretch_point_ERP/2):], data_stretch_load[:int(stretch_point_ERP/2)]))
                    # data_sem_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_sem_load[int(stretch_point_ERP/2):], data_stretch_sem_load[:int(stretch_point_ERP/2)]))

        #### parallelize
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_stretch_data_for_ERP)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))

        #### load data in xr
        xr_data_stretch = xr.DataArray(data=data_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())
        xr_data_sem_stretch = xr.DataArray(data=data_sem_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())

        os.chdir(path_memmap)
        try:
            os.remove(f'data_stretch_ERP.dat')
            del data_stretch_ERP
        except:
            pass

        os.chdir(path_memmap)
        try:
            os.remove(f'data_sem_stretch_ERP.dat')
            del data_sem_stretch_ERP
        except:
            pass

        #### save data
        os.chdir(os.path.join(path_precompute, 'ERP'))
        xr_data_stretch.to_netcdf('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch.to_netcdf('allsujet_ERP_data_sem_stretch.nc')

    return xr_data_stretch, xr_data_sem_stretch





################################################
######## STATS ANALYSIS FUNCTIONS ########
################################################


# data_baseline, data_cond = data_baseline_chan, data_cond_chan
def get_permutation_cluster_1d(data_baseline, data_cond, n_surr):

    if debug:

        colors = {'NM' : 'r', 'PH' : 'b', 'IL' : 'k', 'DL' : 'g'}
        for trial_i in range(data_baseline.shape[0]):
            plt.plot(data_baseline[trial_i,:], color=colors[sujet_list[trial_i][2:4]], label=sujet_list[trial_i][2:4])
        plt.legend()
        plt.show()

        colors = {'NM' : 'r', 'PH' : 'b', 'IL' : 'k', 'DL' : 'g'}
        for trial_i in range(data_cond.shape[0]):
            plt.plot(data_cond[trial_i,:], color=colors[sujet_list[trial_i][2:4]], label=sujet_list[trial_i][2:4])
        plt.legend()
        plt.show()

    n_trials_baselines = data_baseline.shape[0]
    n_trials_cond = data_cond.shape[0]
    n_trials_min = np.array([n_trials_baselines, n_trials_cond]).min()

    data_shuffle = np.concatenate((data_baseline, data_cond), axis=0)
    n_trial_tot = data_shuffle.shape[0]

    test_vec_shuffle = np.zeros((n_surr, data_cond.shape[-1]))

    pixel_based_distrib = np.zeros((n_surr, 2))

    #surr_i = 0
    for surr_i in range(n_surr):

        #### shuffle
        random_sel = np.random.choice(n_trial_tot, size=n_trial_tot, replace=False)
        data_shuffle_baseline = data_shuffle[random_sel[:n_trials_min]]
        data_shuffle_cond = data_shuffle[random_sel[n_trials_min:n_trials_min*2]]

        if debug:
            plt.plot(np.mean(data_shuffle_baseline, axis=0), label='baseline')
            plt.plot(np.mean(data_shuffle_cond, axis=0), label='cond')
            plt.legend()
            plt.show()

            plt.plot(test_vec_shuffle[surr_i,:], label='shuffle')
            plt.hlines(0.05, xmin=0, xmax=data_shuffle.shape[-1], color='r')
            plt.legend()
            plt.show()

        #### extract max min thresh
        _min, _max = np.median(data_shuffle_cond, axis=0).min(), np.median(data_shuffle_cond, axis=0).max()
        # _min, _max = np.percentile(np.median(data_shuffle_cond, axis=0), 1), np.percentile(np.median(data_shuffle_cond, axis=0), 99)
        
        pixel_based_distrib[surr_i, 0] = _min
        pixel_based_distrib[surr_i, 1] = _max

    min, max = np.median(pixel_based_distrib[:,0]), np.median(pixel_based_distrib[:,1]) 
    # min, max = np.percentile(pixel_based_distrib[:,0], 50), np.percentile(pixel_based_distrib[:,1], 50)

    if debug:
        count, _, fig = plt.hist(pixel_based_distrib[:,0], bins=50)
        count, _, fig = plt.hist(pixel_based_distrib[:,1], bins=50)
        plt.vlines([np.median(pixel_based_distrib[:,0]), np.median(pixel_based_distrib[:,1])], ymin=count.min(), ymax=count.max(), color='r')
        plt.show()

        plt.plot(np.mean(data_baseline, axis=0), label='baseline')
        plt.plot(np.mean(data_cond, axis=0), label='cond')
        plt.hlines(min, xmin=0, xmax=data_shuffle.shape[-1], color='r', label='min')
        plt.hlines(max, xmin=0, xmax=data_shuffle.shape[-1], color='r', label='max')
        plt.legend()
        plt.show()

    #### thresh data
    data_thresh = np.mean(data_cond, axis=0).copy()

    _mask = np.logical_or(data_thresh < min, data_thresh > max)
    _mask = _mask*1

    if debug:

        plt.plot(_mask)
        plt.show()

    #### thresh cluster
    mask = np.zeros(data_cond.shape[-1])

    _mask[0], _mask[-1] = 0, 0 # to ensure np.diff detection

    if _mask.sum() != 0:
 
        start, stop = np.where(np.diff(_mask) != 0)[0][::2], np.where(np.diff(_mask) != 0)[0][1::2] 
        
        sizes = stop - start
        min_size = np.percentile(sizes, tf_stats_percentile_cluster_manual_perm)
        if min_size < erp_time_cluster_thresh:
            min_size = erp_time_cluster_thresh
        cluster_signi = sizes >= min_size

        mask = np.zeros(data_cond.shape[-1])

        for cluster_i, cluster_p in enumerate(cluster_signi):

            if cluster_p:

                mask[start[cluster_i]:stop[cluster_i]] = 1

    mask = mask.astype('bool')

    if debug:

        plt.plot(mask)
        plt.show()

    return mask





########################
######## STATS ########
########################



def get_cluster_stats_manual_prem_allsujet(stretch=False):

    if stretch and os.path.exists(os.path.join(path_precompute, 'ERP', 'cluster_stats_allsujet_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        cluster_stats = xr.open_dataarray('cluster_stats_allsujet_stretch.nc')

        return cluster_stats
        
    elif stretch == False and os.path.exists(os.path.join(path_precompute, 'ERP', 'cluster_stats_allsujet.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        cluster_stats = xr.open_dataarray('cluster_stats_allsujet.nc')

        return cluster_stats

    else:

        if stretch:
            time_vec = np.arange(stretch_point_ERP)
            time_vec_stats = time_vec
        else:      
            time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
            time_vec_stats = time_vec <= 0

        cluster_stats = np.zeros((len(chan_list_eeg), time_vec.shape[0]))

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch()
        else:
            xr_data, xr_data_sem = compute_ERP()

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            print(nchan)

            data_baseline = xr_data.loc[:, 'VS', nchan, time_vec_stats].values
            data_cond = xr_data.loc[:, 'CHARGE', nchan, time_vec_stats].values
            
            mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

            if debug:

                data_baseline_plot = xr_data.loc[:, 'VS', nchan, :].values
                data_cond_plot = xr_data.loc[:, 'CHARGE', nchan, :].values

                fig, ax = plt.subplots()
                ax.plot(time_vec, data_baseline_plot.mean(axis=0), label='VS')
                ax.plot(time_vec, data_cond_plot.mean(axis=0), label='CHARGE')
                ax.set_title(nchan)

                ax.fill_between(time_vec, np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).min(), 
                                    np.concatenate((data_baseline.mean(axis=0), data_cond.mean(axis=0))).max(), 
                                    where=np.concatenate((mask.astype('int'), np.zeros((time_vec.size - mask.size)))), alpha=0.3, color='r')

                plt.legend()
                plt.show()

            if stretch:
                cluster_stats[nchan_i,:] = mask
            else:
                cluster_stats[nchan_i,:] = np.concatenate((mask, np.zeros((time_vec.size - mask.size)).astype('bool'))) 

        xr_dict = {'chan' : chan_list_eeg, 'time' : time_vec}
        xr_cluster = xr.DataArray(data=cluster_stats, dims=xr_dict.keys(), coords=xr_dict.values())

        os.chdir(os.path.join(path_precompute, 'ERP'))

        if stretch:
            xr_cluster.to_netcdf('cluster_stats_allsujet_stretch.nc')
        else:
            xr_cluster.to_netcdf('cluster_stats_allsujet.nc')

        return xr_cluster











def get_cluster_stats_manual_prem_subject_wise(stretch=False):

    if stretch and os.path.exists(os.path.join(path_precompute, 'ERP', 'cluster_stats_subjectwise_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        cluster_stats = xr.open_dataarray('cluster_stats_subjectwise_stretch.nc')

        return cluster_stats
        
    elif stretch == False and os.path.exists(os.path.join(path_precompute, 'ERP', 'cluster_stats_subjectwise.nc')):

        os.chdir(os.path.join(path_precompute, 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        cluster_stats = xr.open_dataarray('cluster_stats_subjectwise.nc')

        return cluster_stats

    ######## COMPUTE ########
    else:

        if stretch:
            time_vec = np.arange(stretch_point_ERP)
            time_vec_stats = time_vec
        else:      
            time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
            time_vec_stats = time_vec <= 0

        xr_coords = {'sujet' : sujet_list, 'chan' : chan_list_eeg, 'time' : time_vec}

        ######## INITIATE MEMMAP ########
        os.chdir(path_memmap)
        xr_data = np.memmap(f'cluster_based_perm_allsujet.dat', dtype=np.float32, mode='w+', shape=(len(sujet_list), len(chan_list_eeg), time_vec.size))

        ######## PARALLELIZATION FUNCTION ########
        #sujet_i, sujet = 0, sujet_list[0]
        def get_cluster_based_perm_one_sujet(sujet_i, sujet):
        # for sujet_i, sujet in enumerate(sujet_list):

            ######## COMPUTE ERP ########
            print(f'{sujet} COMPUTE ERP')

            erp_data = {}

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = np.where(chan_list_eeg == 'FC1')[0][0], 'FC1'
            for nchan_i, nchan in enumerate(chan_list_eeg):

                erp_data[nchan] =  {}

                #cond_i, cond = 1, 'CHARGE'
                for cond_i, cond in enumerate(cond_list):

                    #### load
                    data = load_data_sujet(sujet, cond)

                    if stretch:
                        data = load_data_sujet(sujet, cond)[nchan_i,:]
                        data = scipy.signal.detrend(data, type='linear')
                        # data = zscore(data)
                        data_chunk, mean_inspi_ratio = stretch_data(respfeatures[cond], stretch_point_ERP, data, srate)
                        

                    else:
                        inspi_starts = respfeatures[cond]['inspi_index'].values
                        data_chunk = np.zeros((inspi_starts.shape[0], time_vec.size))

                        #### chunk
                        x = data[nchan_i,:]
                        x = scipy.signal.detrend(x, type='linear')
                        x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)
                        x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                        for start_i, start_time in enumerate(inspi_starts):

                            t_start = int(start_time + ERP_time_vec[0]*srate)
                            t_stop = int(start_time + ERP_time_vec[1]*srate)

                            if t_start < 0 or t_stop > x.size:
                                continue

                            x_chunk = x[t_start: t_stop]

                            data_chunk[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()

                    erp_data[nchan][cond] = data_chunk
       
            ######## COMPUTE PERMUTATION ########
            print(f'{sujet} COMPUTE PERMUTATION')

            #chan_i, nchan = 0, chan_list_eeg[0]
            for chan_i, nchan in enumerate(chan_list_eeg):

                data_baseline = erp_data[nchan]['VS'][:,time_vec_stats]
                data_cond = erp_data[nchan]['CHARGE'][:,time_vec_stats]

                mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

                if stretch:
                    xr_data[sujet_i, chan_i, :] = mask
                else:
                    xr_data[sujet_i, chan_i, :] = np.concatenate((mask, np.zeros(time_vec.size - mask.size).astype('bool')))
                
            print(f'{sujet} done')

        ######## PARALLELIZATION COMPUTATION ########
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_cluster_based_perm_one_sujet)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))

        ######## SAVE ########
        xr_cluster_based_perm = xr.DataArray(data=xr_data, dims=xr_coords.keys(), coords=xr_coords.values())

        os.chdir(os.path.join(path_precompute, 'ERP'))

        if stretch:
            xr_cluster_based_perm.to_netcdf('cluster_stats_subjectwise_stretch.nc')
        else:
            xr_cluster_based_perm.to_netcdf('cluster_stats_subjectwise.nc')

        os.chdir(path_memmap)
        try:
            os.remove(f'cluster_based_perm_allsujet.dat')
            del xr_data
        except:
            pass

    return xr_cluster_based_perm
















########################
######## EXTRA ########
########################




def shuffle_data_ERP(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle.mean(0)




def shuffle_data_ERP_linear_based(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle





#baseline_values, cond_values = xr_lm_data.loc[:, 'CO2', 'o', :, 'slope'].values, xr_lm_data.loc[:, 'CO2', '-', :, 'slope'].values
def get_stats_topoplots(baseline_values, cond_values, chan_list_eeg):

    data = {'sujet' : [], 'cond' : [], 'chan' : [], 'value' : []}

    for sujet_i in range(baseline_values.shape[0]):

        for chan_i, chan in enumerate(chan_list_eeg):

            data['sujet'].append(sujet_i)
            data['cond'].append('baseline')
            data['chan'].append(chan)
            data['value'].append(baseline_values[sujet_i, chan_i])

            data['sujet'].append(sujet_i)
            data['cond'].append('cond')
            data['chan'].append(chan)
            data['value'].append(cond_values[sujet_i, chan_i])
    
    df_stats = pd.DataFrame(data)

    mask_signi = np.array((), dtype='bool')

    for chan in chan_list_eeg:

        pval = get_stats_df(df=df_stats.query(f"chan == '{chan}'"), predictor='cond', outcome='value', subject='sujet', design='within')

        if pval < 0.05:
            mask_signi = np.append(mask_signi, True)
        else:
            mask_signi = np.append(mask_signi, False)

    if debug:

        plt.hist(df_stats.query(f"chan == '{chan}' and cond == 'cond'")['value'].values, bins=50)
        plt.hist(df_stats.query(f"chan == '{chan}' and cond == 'baseline'")['value'].values, bins=50)
        plt.show()

    return mask_signi








def get_PPI_count(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'PPI_count_linear_based.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_PPI_count = xr.open_dataarray(f'PPI_count_linear_based.nc')

    else:

        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        time_vec = xr_data['time'].values
        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
        df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

        examinateur_list = ['JG', 'MCN', 'TS']

        xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
        xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

        #examinateur = examinateur_list[2]
        for examinateur in examinateur_list:

            if examinateur == 'JG':

                for sujet in sujet_list:

                    respfeatures = load_respfeatures(sujet)

                    data_chunk_allcond = {}
                    data_value_microV = {}

                    t_start_PPI = ERP_time_vec[0]
                    t_stop_PPI = ERP_time_vec[-1]

                    #cond = 'FR_CV_1'
                    for cond in conditions:

                        data_chunk_allcond[cond] = {}
                        data_value_microV[cond] = {}

                        #odor = odor_list[0]
                        for odor in odor_list:

                            print('compute erp')
                            print(sujet, cond, odor)

                            data_chunk_allcond[cond][odor] = {}
                            data_value_microV[cond][odor] = {}

                            #### load
                            data = load_data_sujet(sujet, cond, odor)
                            data = data[:len(chan_list_eeg),:]

                            respfeatures_i = respfeatures[cond][odor]
                            inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                            #### low pass 45Hz
                            for chan_i, chan in enumerate(chan_list_eeg):
                                data[chan_i,:] = iirfilt(data[chan_i,:], srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):

                                #### chunk
                                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                                data_chunk = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                                x = data[nchan_i,:]

                                x_mean, x_std = x.mean(), x.std()
                                microV_SD = int(x_std*1e6)

                                for start_i, start_time in enumerate(inspi_starts):

                                    t_start = int(start_time + t_start_PPI*srate)
                                    t_stop = int(start_time + t_stop_PPI*srate)

                                    data_chunk[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                                if debug:

                                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                    for inspi_i, _ in enumerate(inspi_starts):

                                        plt.plot(time_vec, data_chunk[inspi_i, :], alpha=0.3)

                                    plt.vlines(0, ymax=data_chunk.max(), ymin=data_chunk.min(), color='k')
                                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                    plt.plot(time_vec, data_chunk.mean(axis=0), color='r')
                                    plt.title(f'{cond} {odor} : {data_chunk.shape[0]}, 3SD : {microV_SD}')
                                    plt.show()

                                #### clean
                                data_stretch_clean = data_chunk[~((data_chunk >= 3) | (data_chunk <= -3)).any(axis=1),:]

                                if debug:

                                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                    for inspi_i, _ in enumerate(inspi_starts):

                                        plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                                    plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                    plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                                    plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                                    plt.show()

                                data_chunk_allcond[cond][odor][nchan] = data_stretch_clean
                                data_value_microV[cond][odor][nchan] = microV_SD

                    #### regroup FR_CV
                    # data_chunk_allcond['VS'] = {}
                    # data_value_microV['VS'] = {}

                    # for odor in odor_list:

                    #     data_chunk_allcond['VS'][odor] = {}
                    #     data_value_microV['VS'][odor] = {}

                    #     #### low pass 45Hz
                    #     for nchan_i, nchan in enumerate(chan_list_eeg):

                    #         data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)
                    #         data_value_microV['VS'][odor][nchan] = data_value_microV['FR_CV_1'][odor][nchan] + data_value_microV['FR_CV_2'][odor][nchan] / 2

                    # data_chunk_allcond['FR_CV_1'] = {}
                    # data_chunk_allcond['FR_CV_2'] = {}

                    # data_value_microV['FR_CV_1'] = {}
                    # data_value_microV['FR_CV_2'] = {}

                    #cond = 'CO2'
                    for cond in conditions:

                        #odor = odor_list[0]
                        for odor in odor_list:

                            print('compute surr')
                            print(sujet, cond, odor)

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data_cond = data_chunk_allcond[cond][odor][nchan]

                                Y = data_cond.mean(axis=0)[time_vec_mask]
                                X = time_vec[time_vec_mask]

                                slope_observed, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                _ERP_surr = np.zeros((ERP_n_surrogate))

                                for surr_i in range(ERP_n_surrogate):

                                    ERP_shuffle = np.zeros(data_cond.shape)

                                    for erp_i in range(data_cond.shape[0]):

                                        cut = np.random.randint(0, data_cond.shape[1], 1)[0]

                                        ERP_shuffle[erp_i,:data_cond[:,cut:].shape[1]] = data_cond[erp_i,cut:]
                                        ERP_shuffle[erp_i,data_cond[:,cut:].shape[1]:] = data_cond[erp_i,:cut]
                                            
                                    surr_i_mean = ERP_shuffle.mean(axis=0)

                                    Y = surr_i_mean[time_vec_mask]
                                    X = time_vec[time_vec_mask]

                                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
                                    
                                    _ERP_surr[surr_i] = slope

                                if debug:

                                    Y = data_cond.mean(0)[time_vec_mask]
                                    X = time_vec[time_vec_mask]
                                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                    plt.plot(time_vec, data_cond.mean(axis=0))
                                    plt.gca().invert_yaxis()
                                    plt.title(nchan)
                                    plt.show()

                                    count, hist, _ = plt.hist(_ERP_surr, bins=50)
                                    plt.gca().invert_xaxis()
                                    plt.vlines(slope, ymin=0, ymax=count.max(), color='r', label='raw')
                                    plt.vlines(np.percentile(_ERP_surr, 5), ymin=0, ymax=count.max(), color='b', label='99')
                                    plt.legend()
                                    plt.title(nchan)
                                    plt.show()

                                if slope_observed < np.percentile(_ERP_surr, 5):

                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = 1

            else:

                #sujet = sujet_list[0]
                for sujet in sujet_list:

                    if sujet in ['28NT']:
                        continue

                    #cond = 'CO2'
                    for cond in conditions:

                        if cond in ['FR_CV_1', 'FR_CV_2']:
                            continue

                        #odor = odor_list[0]
                        for odor in odor_list:

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):

                                if nchan in ['Cz', 'Fz']:

                                    _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_PPI_count.to_netcdf(f'PPI_count_linear_based.nc')

    return xr_PPI_count












########################################
######## ERP RESPONSE PROFILE ########
########################################




def plot_ERP_response_profile(xr_data, xr_data_sem):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    # t_start_PPI = PPI_time_vec[0]
    # t_stop_PPI = PPI_time_vec[1]

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    mask_sujet_rep = []
    for sujet in sujet_list:
        if sujet in sujet_best_list:
            mask_sujet_rep.append(True)
        else:
            mask_sujet_rep.append(False)
    mask_sujet_rep = np.array(mask_sujet_rep)   

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

    mask_frontal = []
    for nchan in chan_list_eeg:
        if nchan in ['Fp1', 'Fz', 'Fp2']:
            mask_frontal.append(True)
        else:
            mask_frontal.append(False)
    mask_frontal = np.array(mask_frontal)

    mask_central = []
    for nchan in chan_list_eeg:
        if nchan in ['C3', 'Cz', 'C4']:
            mask_central.append(True)
        else:
            mask_central.append(False)
    mask_central = np.array(mask_central)

    mask_occipital = []
    for nchan in chan_list_eeg:
        if nchan in ['O1', 'Oz', 'O2']:
            mask_occipital.append(True)
        else:
            mask_occipital.append(False)
    mask_occipital = np.array(mask_occipital)

    mask_temporal = []
    for nchan in chan_list_eeg:
        if nchan in ['T7', 'TP9', 'TP10', 'T8']:
            mask_temporal.append(True)
        else:
            mask_temporal.append(False)
    mask_temporal = np.array(mask_temporal)

    ######## SUJET ########

    dict_time = {'metric' : ['time', 'amp'], 'sujet' : sujet_list, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_time = np.zeros((2, len(sujet_list), len(conditions_diff), len(odor_list), len(chan_list_eeg))) 

    xr_erp_profile = xr.DataArray(data_time, coords=dict_time.values(), dims=dict_time.keys())

    print('SUJET')

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 0, 'CO2'
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values
                    sem = xr_data_sem.loc[sujet, cond, odor, nchan, :].values
                    baseline = xr_data.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                    sem_baseline = xr_data_sem.loc[sujet, 'FR_CV_1', odor, nchan, :].values

                    # if ((data_stretch + sem) < (baseline - sem_baseline)).sum() != 0:

                    ERP_time_i = np.argmax(np.abs(data_stretch - baseline))

                    xr_erp_profile.loc['time', sujet, cond, odor, nchan] = time_vec[ERP_time_i]
                    xr_erp_profile.loc['amp', sujet, cond, odor, nchan] = (data_stretch - baseline)[ERP_time_i]

                    if debug:

                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        plt.plot(time_vec, data_stretch, label=cond, color='r')
                        plt.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        plt.plot(time_vec, baseline, label='VS', color='b')
                        plt.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        plt.gca().invert_yaxis()

                        plt.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                        plt.show()

                        plt.plot(time_vec, data_stretch - baseline, label=cond, color='r')
                        plt.show()

                        plt.plot(time_vec, np.abs(data_stretch - baseline), label=cond, color='r')
                        plt.show()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile.loc['time', :, cond, odor, nchan].values.min())
                scales_val_time['max'].append(xr_erp_profile.loc['time', :, cond, odor, nchan].values.max())

                scales_val_amp['min'].append(xr_erp_profile.loc['amp', :, cond, odor, nchan].values.min())
                scales_val_amp['max'].append(xr_erp_profile.loc['amp', :, cond, odor, nchan].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_time = xr_erp_profile.loc['time', :, cond, odor, nchan].values
                data_amp = xr_erp_profile.loc['amp', :, cond, odor, nchan].values
                
                ax.scatter(data_time[mask_sujet_rep], data_amp[mask_sujet_rep], label='rep')
                ax.scatter(data_time[~mask_sujet_rep], data_amp[~mask_sujet_rep], label='no_rep')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.legend()
        plt.suptitle(f"{nchan}")

        # plt.show()

        fig.savefig(f'{nchan}_allsujet.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    if debug:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            for sujet in sujet_list:

                fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

                #cond_i, cond = 0, 'CO2'
                for cond_i, cond in enumerate(conditions_diff):

                    #odor_i, odor = 2, odor_list[2]
                    for odor_i, odor in enumerate(odor_list):

                        ax = axs[odor_i, cond_i]

                        if odor_i == 0:
                            ax.set_title(cond)
                        if cond_i == 0:
                            ax.set_ylabel(odor)

                        data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values
                        sem = xr_data_sem.loc[sujet, cond, odor, nchan, :].values
                        baseline = xr_data.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                        sem_baseline = xr_data_sem.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                        
                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        ax.plot(time_vec, data_stretch, label=cond, color='r')
                        ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        ax.plot(time_vec, baseline, label='VS', color='b')
                        ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        ax.invert_yaxis()

                        ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                plt.suptitle(f"{sujet}")
                plt.show()

    ######## GROUP ########

    dict_time = {'metric' : ['time', 'amp'], 'group' : sujet_group, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_time = np.zeros((2, len(sujet_group), len(conditions_diff), len(odor_list), len(chan_list_eeg))) 

    xr_erp_profile_group = xr.DataArray(data_time, coords=dict_time.values(), dims=dict_time.keys())

    print('GROUP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #group = sujet_group[0]
    for group in sujet_group:

        print(group)

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 0, 'CO2'
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_best_list, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    # if ((data_stretch + sem) < (baseline - sem_baseline)).sum() != 0:

                    ERP_time_i = np.argmax(np.abs(data_stretch - baseline))

                    xr_erp_profile_group.loc['time', group, cond, odor, nchan] = time_vec[ERP_time_i]
                    xr_erp_profile_group.loc['amp', group, cond, odor, nchan] = (data_stretch - baseline)[ERP_time_i]

                    if debug:

                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        plt.plot(time_vec, data_stretch, label=cond, color='r')
                        plt.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        plt.plot(time_vec, baseline, label='VS', color='b')
                        plt.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        plt.gca().invert_yaxis()

                        plt.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                        plt.show()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        for group in sujet_group:

            #cond_i, cond = 2, conditions_diff[2]
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    scales_val_time['min'].append(xr_erp_profile_group.loc['time', :, cond, odor, nchan].values.min())
                    scales_val_time['max'].append(xr_erp_profile_group.loc['time', :, cond, odor, nchan].values.max())

                    scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values.min())
                    scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_time = xr_erp_profile_group.loc['time', :, cond, odor, nchan].values
                data_amp = xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values
                
                ax.scatter(xr_erp_profile_group.loc['time', 'allsujet', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'allsujet', cond, odor, nchan].values, label='allsujet')
                ax.scatter(xr_erp_profile_group.loc['time', 'rep', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'rep', cond, odor, nchan].values, label='rep')
                ax.scatter(xr_erp_profile_group.loc['time', 'non_rep', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'non_rep', cond, odor, nchan].values, label='non_rep')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.legend()
        plt.suptitle(f"{nchan}")

        # plt.show()

        fig.savefig(f'{nchan}_allgroup.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()       

    #### plot allgroup allchan
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    scales_val_time = {'min' : [], 'max' : []}
    scales_val_amp = {'min' : [], 'max' : []}

    #cond_i, cond = 2, conditions_diff[2]
    for cond_i, cond in enumerate(conditions_diff):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            scales_val_time['min'].append(xr_erp_profile_group.loc['time', :, cond, odor, :].values.min())
            scales_val_time['max'].append(xr_erp_profile_group.loc['time', :, cond, odor, :].values.max())

            scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', :, cond, odor, :].values.min())
            scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', :, cond, odor, :].values.max())

    min_time = np.array(scales_val_time['min']).min()
    max_time = np.array(scales_val_time['max']).max()

    min_amp = np.array(scales_val_amp['min']).min()
    max_amp = np.array(scales_val_amp['max']).max()

    #cond_i, cond = 0, 'CO2'
    for cond_i, cond in enumerate(conditions_diff):

        #odor_i, odor = 2, odor_list[2]
        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i, cond_i]

            if odor_i == 0:
                ax.set_title(cond)
            if cond_i == 0:
                ax.set_ylabel(odor)
            
            ax.scatter(xr_erp_profile_group.loc['time', 'allsujet', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'allsujet', cond, odor, :].values, label='allsujet')
            ax.scatter(xr_erp_profile_group.loc['time', 'rep', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'rep', cond, odor, :].values, label='rep')
            ax.scatter(xr_erp_profile_group.loc['time', 'non_rep', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'non_rep', cond, odor, :].values, label='non_rep')

            ax.set_ylim(min_amp, max_amp)
            ax.set_xlim(min_time, max_time)

            ax.invert_yaxis()

            ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
            ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

    plt.suptitle(f"{group}")
    plt.legend()

    # plt.show()

    fig.savefig(f'{nchan}_allgroup.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()       

    #### plot allchan
    for group in sujet_group:

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.min())
                scales_val_time['max'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.max())

                scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.min())
                scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)
                
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_frontal], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_frontal], label='frontal')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_central], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_central], label='central')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_occipital], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_occipital], label='occipital')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_temporal], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_temporal], label='temporal')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.suptitle(f"{group}")
        plt.legend()

        # plt.show()

        fig.savefig(f'allchan_topo_{group}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()   



        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.min())
                scales_val_time['max'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.max())

                scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.min())
                scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)
                
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values, xr_erp_profile_group.loc['amp', group, cond, odor, :].values)

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.suptitle(f"{group}")

        # plt.show()

        fig.savefig(f'allchan_allchan_{group}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()    














########################################
######## ERP RESPONSE STATS ########
########################################



def plot_erp_response_stats(xr_data):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    # t_start_PPI = PPI_time_vec[0]
    # t_stop_PPI = PPI_time_vec[1]

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

    sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])

    #### generate df
    # time_vec_mask = xr_data['time'].values[(time_vec >= -1) & (time_vec <= 0)]
    time_vec_mask = xr_data['time'].values[(time_vec >= -2) & (time_vec <= 2)]

    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].sum('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)

    df_min = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    df_max = xr_data.loc[:, :, :, :, time_vec_mask].max('time').to_dataframe(name='val').reset_index(drop=False)

    df_minmax = df_min.copy()
    df_minmax['val'] = np.abs(df_min['val'].values) + np.abs(df_max['val'].values)

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    predictor = 'cond'
    outcome = 'val'

    for group in sujet_group:

        for nchan in chan_list_eeg:

            fig, axs = plt.subplots(ncols=len(odor_list))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                ax.set_ylabel(odor)

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}'")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")

                ax = auto_stats(df_stats, predictor, outcome, ax=ax, subject='sujet', design='within', mode='box', transform=False, verbose=False)
                
            plt.suptitle(f"{odor_list} {nchan} {group}")
            plt.tight_layout()

            # plt.show()

            fig.savefig(f'stats_inter_{nchan}_{group}.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()    

    predictor = 'odor'
    outcome = 'val'

    for group in sujet_group:

        for nchan in chan_list_eeg:

            fig, axs = plt.subplots(ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                ax = axs[cond_i]

                ax.set_ylabel(odor)

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}'")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")

                ax = auto_stats(df_stats, predictor, outcome, ax=ax, subject='sujet', design='within', mode='box', transform=False, verbose=False)
                
            plt.suptitle(f"{conditions} {nchan} {group}")
            plt.tight_layout()

            # plt.show()

            fig.savefig(f'stats_intra_{nchan}_{group}.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()  




def get_df_stats(xr_data):

    if os.path.exists(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'df_stats_all_intra.xlsx')):

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff'))

        print('ALREADY COMPUTED', flush=True)

        df_stats_all_intra = pd.read_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter = pd.read_excel('df_stats_all_inter.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter}

    else:

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        # t_start_PPI = PPI_time_vec[0]
        # t_stop_PPI = PPI_time_vec[1]

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        sujet_group = ['allsujet', 'rep', 'non_rep']

        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
        mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

        conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

        #### generate df
        # time_vec_mask = xr_data['time'].values[(time_vec >= -1) & (time_vec <= 0)]
        time_vec_mask = xr_data['time'].values[(time_vec >= -2) & (time_vec <= 2)]

        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].sum('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)

        df_min = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        df_max = xr_data.loc[:, :, :, :, time_vec_mask].max('time').to_dataframe(name='val').reset_index(drop=False)

        df_minmax = df_min.copy()
        df_minmax['val'] = np.abs(df_min['val'].values) + np.abs(df_max['val'].values)

        #### plot
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

        #### df generation
        predictor = 'odor'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            for nchan_i, nchan in enumerate(chan_list_eeg):

                #cond_i, cond = 2, conditions[2]
                for cond_i, cond in enumerate(conditions):

                    if group == 'allsujet':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}'")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}'")
                    if group == 'rep':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list_rev.tolist()}")
                    if group == 'non_rep':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
        
                    if group_i + nchan_i + cond_i == 0:
                        df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'nchan', np.array([nchan]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'cond', np.array([cond]*df_stats_all.shape[0]))

                    else:
                        _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'nchan', np.array([nchan]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'cond', np.array([cond]*_df_stats_all.shape[0]))
                        df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['inter'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_inter = df_stats_all.copy()

        predictor = 'cond'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            for nchan_i, nchan in enumerate(chan_list_eeg):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}'")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}'")
                    if group == 'rep':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list_rev.tolist()}")
                    if group == 'non_rep':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
        
                    if group_i + nchan_i + odor_i == 0:
                        df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'nchan', np.array([nchan]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'odor', np.array([odor]*df_stats_all.shape[0]))

                    else:
                        _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'nchan', np.array([nchan]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'odor', np.array([odor]*_df_stats_all.shape[0]))
                        df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['intra'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_intra = df_stats_all.copy()

        df_stats_all_intra = df_stats_all_intra.query(f"A == 'FR_CV_1' or B == 'FR_CV_1'")
        df_stats_all_inter = df_stats_all_inter.query(f"A == 'o' or B == 'o'")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff'))

        df_stats_all_intra.to_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter.to_excel('df_stats_all_inter.xlsx')

        df_stats_all_intra.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'FR_CV_1' or B == 'FR_CV_1'").to_excel('df_stats_all_intra_signi.xlsx')
        df_stats_all_inter.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'o' or B == 'o'").to_excel('df_stats_all_inter_signi.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter}

    return df_stats_all









def get_cluster_stats(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats.pkl')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        with open('cluster_stats.pkl', 'rb') as fp:
            cluster_stats = pickle.load(fp)

        with open('cluster_stats_rep_norep.pkl', 'rb') as fp:
            cluster_stats_rep_norep = pickle.load(fp)


    else:

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        mat_adjacency = mne.channels.find_ch_adjacency(info, 'eeg')[0]

        conditions_diff = ['MECA', 'CO2', 'FR_CV_2']
        odor_diff = ['+', '-']
        sujet_group = ['allsujet', 'rep', 'non_rep']

        cluster_stats = {}

        cluster_stats['intra'] = {}

        #group = sujet_group[0]
        for group in sujet_group:

            cluster_stats['intra'][group] = {}

            #nchan = chan_list_eeg[0]
            for nchan in chan_list_eeg:

                print(group, nchan)

                cluster_stats['intra'][group][nchan] = {}

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    cluster_stats['intra'][group][nchan][odor] = {}

                    #cond = conditions_diff[0]
                    for cond in conditions_diff:

                        cluster_stats['intra'][group][nchan][odor][cond] = {}

                        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values

                        if debug:

                            plt.plot(data_baseline.mean(axis=0))
                            plt.plot(data_baseline.mean(axis=0))
                            plt.show()

                        # n_conditions = 2
                        # n_observations = data_cond.shape[0]
                        # pval = 0.05  # arbitrary
                        # dfn = n_conditions - 1  # degrees of freedom numerator
                        # dfd = n_observations - 2  # degrees of freedom denominator
                        # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                        # thresh = int(np.round(thresh))
                        
                        # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                        #     [data_baseline, data_cond],
                        #     n_permutations=1000,
                        #     threshold=None,
                        #     tail=1,
                        #     n_jobs=4,
                        #     out_type="mask",
                        #     verbose='CRITICAL'
                        # )

                        data_diff = data_baseline-data_cond

                        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                        cluster_stats['intra'][group][nchan][odor][cond]['cluster'] = clusters
                        cluster_stats['intra'][group][nchan][odor][cond]['pval'] = cluster_p_values

        cluster_stats['inter'] = {}

        for group in sujet_group:

            cluster_stats['inter'][group] = {}

            for nchan in chan_list_eeg:

                print(group, nchan)

                cluster_stats['inter'][group][nchan] = {}

                #cond_i, cond = 2, conditions[2]
                for cond_i, cond in enumerate(conditions):

                    cluster_stats['inter'][group][nchan][cond] = {}

                    for odor in odor_diff:

                        cluster_stats['inter'][group][nchan][cond][odor] = {}

                        data_baseline = xr_data.loc[:, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values

                        # n_conditions = 2
                        # n_observations = data_cond.shape[0]
                        # pval = 0.05  # arbitrary
                        # dfn = n_conditions - 1  # degrees of freedom numerator
                        # dfd = n_observations - 2  # degrees of freedom denominator
                        # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                        # thresh = int(np.round(thresh))
                        
                        # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                        #     [data_baseline, data_cond],
                        #     n_permutations=1000,
                        #     threshold=thresh,
                        #     tail=1,
                        #     n_jobs=4,
                        #     out_type="mask",
                        #     verbose='CRITICAL'
                        # )

                        data_diff = data_baseline-data_cond

                        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                        cluster_stats['inter'][group][nchan][cond][odor]['cluster'] = clusters
                        cluster_stats['inter'][group][nchan][cond][odor]['pval'] = cluster_p_values

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        
        with open('cluster_stats.pkl', 'wb') as fp:
            pickle.dump(cluster_stats, fp)

        cluster_stats_rep_norep = {}
        sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])
        sujet_best_list_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])

        for nchan in chan_list_eeg:

            print(nchan)

            cluster_stats_rep_norep[nchan] = {}

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                cluster_stats_rep_norep[nchan][odor] = {}

                for cond in conditions:

                    cluster_stats_rep_norep[nchan][odor][cond] = {}

                    data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    # n_conditions = 2
                    # n_observations = data_cond.shape[0]
                    # pval = 0.05  # arbitrary
                    # dfn = n_conditions - 1  # degrees of freedom numerator
                    # dfd = n_observations - 2  # degrees of freedom denominator
                    # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                    # thresh = int(np.round(thresh))
                    
                    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                    #     [data_baseline, data_cond],
                    #     n_permutations=1000,
                    #     threshold=thresh,
                    #     tail=1,
                    #     n_jobs=4,
                    #     out_type="mask",
                    #     verbose='CRITICAL'
                    # )

                    n_obs_min = np.array([data_baseline.shape[0], data_cond.shape[0]]).min()

                    baseline_sel = np.random.choice(range(data_baseline.shape[0]), size=n_obs_min)
                    cond_sel = np.random.choice(range(data_cond.shape[0]), size=n_obs_min)

                    data_diff = data_baseline[baseline_sel,:] - data_cond[cond_sel,:]

                    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                    cluster_stats_rep_norep[nchan][odor][cond]['cluster'] = clusters
                    cluster_stats_rep_norep[nchan][odor][cond]['pval'] = cluster_p_values

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        
        with open('cluster_stats_rep_norep.pkl', 'wb') as fp:
            pickle.dump(cluster_stats_rep_norep, fp)

    return cluster_stats, cluster_stats_rep_norep


def get_permutation_cluster_1d_stretch_one_cond(data_cond, x, respfeatures_stretch, n_surr):

    cycles_length = respfeatures_stretch[['inspi_index', 'expi_index', 'next_inspi_index']].diff(axis=1)[['expi_index', 'next_inspi_index']].values

    n_trials_cond = cycles_length.shape[0]

    respfeatures_surr = respfeatures_stretch.copy()
    start_inspi_init = respfeatures_stretch['inspi_index'].values[0]

    surr_cycles = np.zeros((n_trials_cond, 3), dtype='int')
    surr_cycles[0,0] = start_inspi_init 
    surr_erp_data = np.zeros((n_trials_cond, data_cond.shape[-1]))
    surr_erp_data_median = np.zeros((n_surr, data_cond.shape[-1]))

    #surr_i = 0
    for surr_i in range(n_surr):

        #### shuffle
        shuffle = np.random.choice(np.arange(n_trials_cond), size=n_trials_cond, replace=False)
        cycles_length_shuffled = cycles_length[shuffle,:]

        for cycle_i in range(cycles_length_shuffled.shape[0]):

            if cycle_i == n_trials_cond-1:

                surr_cycles[cycle_i,1] = surr_cycles[cycle_i,0] + cycles_length_shuffled[cycle_i,0]
                surr_cycles[cycle_i,2] = surr_cycles[cycle_i,1] + cycles_length_shuffled[cycle_i,1]

            else:

                surr_cycles[cycle_i,1] = surr_cycles[cycle_i,0] + cycles_length_shuffled[cycle_i,0]
                surr_cycles[cycle_i,2] = surr_cycles[cycle_i,1] + cycles_length_shuffled[cycle_i,1]
                surr_cycles[cycle_i+1,0] = surr_cycles[cycle_i,2]


        respfeatures_surr.iloc[:,1], respfeatures_surr.iloc[:,2], respfeatures_surr.iloc[:,3] = surr_cycles[:,0], surr_cycles[:,1], surr_cycles[:,2]
        respfeatures_surr.iloc[:,4], respfeatures_surr.iloc[:,5], respfeatures_surr.iloc[:,6] = respfeatures_surr.iloc[:,1]/srate, respfeatures_surr.iloc[:,2]/srate, respfeatures_surr.iloc[:,3]/srate

        surr_erp_data, mean_inspi_ratio = stretch_data(respfeatures_surr, stretch_point_ERP, x, srate)

        if debug:
            for i in range(n_trials_cond):
                plt.plot(surr_erp_data[i,:], alpha=0.4)
            plt.plot(surr_erp_data.mean(axis=0), color='r')
            plt.show()

            plt.plot(data_cond.mean(axis=0), label='cond')
            plt.plot(surr_erp_data.mean(axis=0), label='surr')
            plt.legend()
            plt.show()
            
        #### inverse to have inspi on the right and expi on the left
        # surr_erp_data = np.hstack((surr_erp_data[:,int(stretch_point_ERP/2):], surr_erp_data[:,:int(stretch_point_ERP/2)]))
        
        #### export data
        surr_erp_data_median[surr_i,:] = np.mean(surr_erp_data, axis=0)

        if debug:

            plt.plot(np.mean(surr_erp_data, axis=0))
            plt.show()

    # min, max = np.median(pixel_based_distrib[:,0,:], axis=0), np.median(pixel_based_distrib[:,1,:], axis=0) 
    min, max = np.percentile(surr_erp_data_median, 1, axis=0), np.percentile(surr_erp_data_median, 99, axis=0)

    if debug:
        plt.plot(min, color='r')
        plt.plot(max, color='r')
        plt.plot(np.mean(data_cond, axis=0), color='g', label='data')
        plt.plot(np.mean(surr_erp_data_median, axis=0), color='b', label='surr')
        plt.legend()
        plt.show()

        for i in range(400):

            plt.plot(surr_erp_data_median[i,:], alpha=0.3)

        plt.plot(data_cond.mean(axis=0), color='r', label='cond')
        plt.show()

    #### thresh data
    data_thresh = np.mean(data_cond, axis=0).copy()

    _mask = np.logical_or(data_thresh < min, data_thresh > max)
    _mask = _mask*1

    if debug:

        plt.plot(_mask)
        plt.show()

    #### thresh cluster
    mask = np.zeros(data_cond.shape[-1])

    _mask[0], _mask[-1] = 0, 0 # to ensure np.diff detection

    if _mask.sum() != 0:
 
        start, stop = np.where(np.diff(_mask) != 0)[0][::2], np.where(np.diff(_mask) != 0)[0][1::2] 
        
        sizes = stop - start
        med, mad = np.median(sizes), int(np.median(np.abs(np.median(sizes) - sizes)) / 0.6744897501960817)
        min_size = med + mad
        
        if min_size < erp_time_cluster_thresh:
            min_size = erp_time_cluster_thresh
        cluster_signi = sizes >= min_size

        mask = np.zeros(data_cond.shape[-1])

        for cluster_i, cluster_p in enumerate(cluster_signi):

            if cluster_p:

                mask[start[cluster_i]:stop[cluster_i]] = 1

    mask = mask.astype('bool')

    if debug:

        plt.plot(mask)
        plt.show()

    return mask





# data_baseline, data_cond = data_baseline_chan, data_cond_chan
def get_permutation_cluster_1d_one_cond(data_cond, x, n_surr):

    n_trials_cond = data_cond.shape[0]

    surr_erp_data = np.zeros((n_trials_cond*2, data_cond.shape[-1]))
    surr_erp_data_median = np.zeros((n_surr, data_cond.shape[-1]))

    pixel_based_distrib = np.zeros((n_surr, 2, data_cond.shape[-1]))

    #surr_i = 0
    for surr_i in range(n_surr):

        #### shuffle
        seeds = np.random.randint(low=0, high=x.size-data_cond.shape[-1], size=n_trials_cond*2)

        if debug:

            plt.plot(x)
            plt.vlines(seeds, ymin=x.min(), ymax=x.max(), color='r')
            plt.show()

        for seed_i, seed in enumerate(seeds):
            t_start = seed
            t_stop = seed+data_cond.shape[-1]
            x_chunk = x[t_start:t_stop]

            surr_erp_data[seed_i,:] = (x_chunk - x_chunk.mean()) / x_chunk.std()

        surr_erp_data_clean = surr_erp_data[((surr_erp_data <= -3) | (surr_erp_data >= 3)).sum(axis=1) == 0]
        surr_erp_data_clean = surr_erp_data_clean[:n_trials_cond,:]

        if debug:
            for i in range(n_trials_cond):
                plt.plot(surr_erp_data_clean[i,:])
            plt.show()

            plt.plot(data_cond.mean(axis=0), label='cond')
            plt.plot(surr_erp_data_clean.mean(axis=0), label='surr')
            plt.legend()
            plt.show()
            
        surr_erp_data_median[surr_i,:] = np.mean(surr_erp_data_clean, axis=0)

    # min, max = np.median(pixel_based_distrib[:,0,:], axis=0), np.median(pixel_based_distrib[:,1,:], axis=0) 
    min, max = np.percentile(surr_erp_data_median, 1, axis=0), np.percentile(surr_erp_data_median, 99, axis=0)

    if debug:
        plt.plot(min, color='r')
        plt.plot(max, color='r')
        plt.plot(np.mean(data_cond, axis=0), color='g')
        plt.plot(np.mean(surr_erp_data_median, axis=0), color='g')
        plt.show()
        
        count, _, fig = plt.hist(pixel_based_distrib[:,0], bins=50)
        count, _, fig = plt.hist(pixel_based_distrib[:,1], bins=50)
        plt.vlines([np.median(pixel_based_distrib[:,0]), np.median(pixel_based_distrib[:,1])], ymin=count.min(), ymax=count.max(), color='r')
        plt.show()

        plt.plot(np.mean(data_cond, axis=0), label='cond')
        plt.hlines(min, xmin=0, xmax=data_cond.shape[-1], color='r', label='min')
        plt.hlines(max, xmin=0, xmax=data_cond.shape[-1], color='r', label='max')
        plt.legend()
        plt.show()

        for i in range(400):

            seeds = np.random.randint(low=0, high=x.size-data_cond.shape[-1], size=n_trials_cond*2)

            for seed_i, seed in enumerate(seeds):
                t_start = seed
                t_stop = seed+data_cond.shape[-1]
                x_chunk = x[t_start:t_stop]

                surr_erp_data[seed_i,:] = (x_chunk - x_chunk.mean()) / x_chunk.std()

            surr_erp_data_clean = surr_erp_data[((surr_erp_data <= -3) | (surr_erp_data >= 3)).sum(axis=1) == 0]
            surr_erp_data_clean = surr_erp_data_clean[:n_trials_cond,:]

            plt.plot(surr_erp_data_clean.mean(axis=0), alpha=0.3)

        plt.plot(data_cond.mean(axis=0), color='r', label='cond')
        plt.show()

    #### thresh data
    data_thresh = np.mean(data_cond, axis=0).copy()

    _mask = np.logical_or(data_thresh < min, data_thresh > max)
    _mask = _mask*1

    if debug:

        plt.plot(_mask)
        plt.show()

    #### thresh cluster
    mask = np.zeros(data_cond.shape[-1])

    _mask[0], _mask[-1] = 0, 0 # to ensure np.diff detection

    if _mask.sum() != 0:
 
        start, stop = np.where(np.diff(_mask) != 0)[0][::2], np.where(np.diff(_mask) != 0)[0][1::2] 
        
        sizes = stop - start
        med, mad = np.median(sizes), int(np.median(np.abs(np.median(sizes) - sizes)) / 0.6744897501960817)
        min_size = med + mad
        
        if min_size < erp_time_cluster_thresh:
            min_size = erp_time_cluster_thresh
        cluster_signi = sizes >= min_size

        mask = np.zeros(data_cond.shape[-1])

        for cluster_i, cluster_p in enumerate(cluster_signi):

            if cluster_p:

                mask[start[cluster_i]:stop[cluster_i]] = 1

    mask = mask.astype('bool')

    if debug:

        plt.plot(mask)
        plt.show()

    return mask








################################################
######## GENERATE PPI EVALUATION ########
################################################



def generate_ppi_evaluation(xr_data):

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    #### generate parameters
    dict = {'sujet' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'indice' : [], 'PPI' : []}

    chan_list_blind_evaluation = ['Cz', 'Fz']

    for sujet in sujet_list:

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                for nchan_i, nchan in enumerate(chan_list_blind_evaluation):

                    dict['sujet'].append(sujet)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['indice'].append(0)
                    dict['PPI'].append(0)

    df_PPI_blind_evaluation = pd.DataFrame(dict)
    df_PPI_blind_evaluation = df_PPI_blind_evaluation.sample(frac=1).reset_index(drop=True)
    df_PPI_blind_evaluation['indice'] = np.arange(df_PPI_blind_evaluation.shape[0])+1

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_PPI_blind_evaluation.to_excel('df_PPI_blind_evaluation.xlsx')

    ######## SUMMARY NCHAN ########

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))

    for indice in range(df_PPI_blind_evaluation.shape[0]):

        print_advancement(indice, df_PPI_blind_evaluation.shape[0], [25, 50, 75])

        fig, ax = plt.subplots()

        plt.suptitle(indice+1)

        fig.set_figheight(10)
        fig.set_figwidth(10)

        sujet, cond, odor, nchan = df_PPI_blind_evaluation.iloc[indice].values[0], df_PPI_blind_evaluation.iloc[indice].values[1], df_PPI_blind_evaluation.iloc[indice].values[2], df_PPI_blind_evaluation.iloc[indice].values[3]

        data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values

        ax.set_ylim(-3, 3)

        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

        ax.plot(time_vec, data_stretch)
        # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
        # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

        ax.invert_yaxis()

        ax.vlines(0, ymin=-3, ymax=3, colors='g')  
        ax.hlines(0, xmin=PPI_time_vec[0], xmax=PPI_time_vec[-1], colors='g') 

        # plt.show()

        #### save
        fig.savefig(f'{indice+1}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()













########################################
######## RESPI ERP ANALYSIS ########
########################################






def plot_mean_respi():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'allsujet_ERP_respi.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('xr_respi ALREADY COMPUTED', flush=True)

        xr_respi = xr.open_dataarray(f'allsujet_ERP_respi.nc')
        xr_itl = xr.open_dataarray(f'allsujet_ERP_respi_itl.nc')

    else:

        xr_respi = get_data_erp_respi()
        xr_itl = get_data_itl()

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_respi.to_netcdf('allsujet_ERP_respi.nc')
        xr_itl.to_netcdf('allsujet_ERP_respi_itl.nc')

    t_start_PPI = -4
    t_stop_PPI = 4

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    max = np.array([(xr_itl.mean('sujet').values*-1).max(), (xr_respi.mean('sujet').values).max()]).max()
    min = np.array([(xr_itl.mean('sujet').values*-1).min(), (xr_respi.mean('sujet').min().values).min()]).min()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        ax.set_title(f"{odor}")

        ax.plot(time_vec, xr_itl.mean('sujet').loc['VS',:].values*-1, label='VS_itl', color='g', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['CO2',:].values*-1, label='CO2_itl', color='r', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['ITL',:].values*-1, label='ITL_itl', color='b', linestyle=':', dashes=(5, 10))

        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'VS_1', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_2', :].mean('sujet'), label=f'VS_2', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'MECA', :].mean('sujet'), label=f'MECA', color='b')

        ax.vlines(0, ymin=min, ymax=max, color='k')

    plt.legend()
    
    plt.suptitle(f"comparison ITL")

    # plt.show()

    plt.savefig(f"allsujet_ERP_comparison_ITL.png")
    plt.close('all')


    #### plot sujet respi
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))
    sujet_group = ['allsujet', 'rep', 'no_rep']

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(conditions), figsize=(15,10))

        for cond_i, cond in enumerate(conditions):

            ax = axs[cond_i]

            ax.set_title(f"{cond}")

            ax.plot(time_vec, xr_data.loc[:, 'o', cond, :].mean('sujet'), label=f'o', color='b')
            ax.plot(time_vec, xr_data.loc[:, '-', cond, :].mean('sujet'), label=f'-', color='r')
            ax.plot(time_vec, xr_data.loc[:, '+', cond, :].mean('sujet'), label=f'+', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_COND_mean_{group}.png")
        
        plt.close('all')

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]

            ax.set_title(f"{odor}")

            ax.plot(time_vec, xr_data.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'FR_CV_1', color='c')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'FR_CV_2', :].mean('sujet'), label=f'FR_CV_2', color='b')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'MECA', :].mean('sujet'), label=f'MECA', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_ODOR_mean_{group}.png")
        
        plt.close('all')

    if debug:

        cond = 'FR_CV_1'
        odor = 'o'

        for sujet in sujet_list:

            plt.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=str(sujet.data), alpha=0.25)

        plt.plot(time_vec, xr_data.loc[:, odor, cond, :].mean('sujet'), label=str(sujet.data), alpha=1, color='r')
        plt.vlines(0, ymin=xr_data.loc[:, odor, cond, :].min(), ymax=xr_data.loc[:, odor, cond, :].max(), color='k')
        plt.title(f'allsujet {cond} {odor}: {sujet_list.shape[0]}')
        # plt.legend()
        plt.show()

        for sujet in sujet_list:

            fig, axs = plt.subplots(ncols=len(odor_list))

            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                for cond in conditions:

                    ax.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=cond)

            plt.legend()
            plt.suptitle(sujet)
            plt.show()

        plt.plot(xr_data.loc[:, 'o', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '+', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '-', cond, :].mean('sujet'))













################################
######## PPI PROPORTION ########
################################


def plot_PPI_proportion(xr_PPI_count):

    cond_temp = ['VS', 'CO2', 'MECA']

    group_list = ['allsujet', 'rep', 'no_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    #### MCN, TS

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

    examinateur_list = ['MCN', 'TS']

    xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    _xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

    for examinateur in examinateur_list:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            if sujet in ['28NT']:
                continue

            #cond = 'CO2'
            for cond in conditions:

                if cond in ['FR_CV_1', 'FR_CV_2']:
                    continue

                #odor = odor_list[0]
                for odor in odor_list:

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        if nchan in ['Cz', 'Fz']:

                            _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                            _xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

    df_PPI_count = _xr_PPI_count.to_dataframe(name='PPI').reset_index()

    dict = {'examinateur' : [], 'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for examinateur in examinateur_list:

        for group in group_list:

            #cond = 'VS'
            for cond in cond_temp:

                for odor in odor_list:

                    for nchan in chan_list_eeg:

                        if group == 'allsujet':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'no_rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        
                        if df_plot['PPI'].sum() == 0:
                            prop = 0
                        else:
                            prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                        dict['examinateur'].append(examinateur)
                        dict['group'].append(group)
                        dict['cond'].append(cond)
                        dict['odor'].append(odor)
                        dict['nchan'].append(nchan)
                        dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for examinateur in examinateur_list:

        for nchan in ['Cz', 'Fz']:

            df_plot = df_PPI_plot.query(f"examinateur == '{examinateur}' and nchan == '{nchan}'")
            sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
            plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
            plt.ylim(0,100)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{nchan}_{examinateur}.png")
            plt.close('all')

    #### JG

    df_PPI_count = xr_PPI_count.to_dataframe(name='PPI').reset_index()
    df_PPI_count = df_PPI_count.query(f"examinateur == 'JG'")

    dict = {'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for group in group_list:

        #cond = 'VS'
        for cond in conditions:

            for odor in odor_list:

                for nchan in chan_list_eeg:

                    if group == 'allsujet':
                        df_plot = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'no_rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    
                    if df_plot['PPI'].sum() == 0:
                        prop = 0
                    else:
                        prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                    dict['group'].append(group)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for nchan in chan_list_eeg:

        df_plot = df_PPI_plot.query(f"nchan == '{nchan}'")
        sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
        plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
        plt.ylim(0,100)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{nchan}_JG.png")
        plt.close('all')




    







################################
######## PERMUTATION ########
################################




def compute_topoplot_stats_allsujet_perm(xr_data, perm_type):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    times = xr_data['time'].values

    ######## INTRA ########
    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            min = np.append(min, data_perm_topo.min())
            max = np.append(max, data_perm_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            print(perm_type, 'intra', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                if perm_type == 'mne':

                    data_perm = data_cond_chan - data_baseline_chan 

                    T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                        data_perm,
                        n_permutations=1000,
                        threshold=None,
                        tail=0,
                        n_jobs=4,
                        out_type="mask",
                        verbose=False
                    )

                    if (clusters_p_values < 0.05).any():
                        
                        mask_signi[chan_i] = True

                else:

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                    if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                        mask_signi[chan_i] = True 

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'ALLSUJET INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"perm_{perm_type}_intra_allsujet.jpeg")

    plt.close('all')
    
    # plt.show()

    ######## INTER ########
    #### scale
    min = np.array([])
    max = np.array([])

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            min = np.append(min, data_perm_topo.min())
            max = np.append(max, data_perm_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            print(perm_type, 'inter', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_perm = data_cond_red - data_baseline_red 
            data_perm_topo = data_perm.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                if perm_type == 'mne':

                    data_perm = data_cond_chan - data_baseline_chan 

                    T_obs, clusters, clusters_p_values, H0 = permutation_cluster_1samp_test(
                        data_perm,
                        n_permutations=1000,
                        threshold=None,
                        tail=0,
                        n_jobs=4,
                        out_type="mask",
                        verbose=False
                    )

                    if (clusters_p_values < 0.05).any():
                            
                        mask_signi[chan_i] = True

                else:

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                    if perm_vec.sum() >= int(erp_time_cluster_thresh*1e-3*srate): 
                        mask_signi[chan_i] = True 
            
            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_perm_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim), cmap='seismic')

    plt.tight_layout()

    plt.suptitle(f'ALLSUJET INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"perm_{perm_type}_inter_allsujet.jpeg")

    plt.close('all')

    # plt.show()






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    ######## ERP ########

    compute_ERP()
    compute_ERP_stretch()
    
    ######## STATS ########
    
    get_cluster_stats_manual_prem_allsujet(stretch=False)
    get_cluster_stats_manual_prem_allsujet(stretch=True)

    get_cluster_stats_manual_prem_subject_wise(stretch=False)
    get_cluster_stats_manual_prem_subject_wise(stretch=True)


