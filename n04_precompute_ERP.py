
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
            
            mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate, stat_design=stat_design, mode_grouped=mode_grouped, 
                                              mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                              size_thresh_alpha=size_thresh_alpha)

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

                mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate, stat_design=stat_design, mode_grouped=mode_grouped, 
                                              mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                              size_thresh_alpha=size_thresh_alpha)

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

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate, stat_design=stat_design, mode_grouped=mode_grouped, 
                                              mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                              size_thresh_alpha=size_thresh_alpha)

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

                    perm_vec = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate, stat_design=stat_design, mode_grouped=mode_grouped, 
                                              mode_generate_surr=mode_generate_surr_1d, mode_select_thresh=mode_select_thresh_1d, percentile_thresh=percentile_thresh, 
                                              size_thresh_alpha=size_thresh_alpha)

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


