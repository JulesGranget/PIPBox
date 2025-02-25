
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


