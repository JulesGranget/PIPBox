

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
def get_MI_allsujet(stretch):

    #### verify computation
    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'FC', f'MI_allsujet_stretch.nc')):
            print(f'ALREADY DONE MI STRETCH')
            return

    else:

        if os.path.exists(os.path.join(path_precompute, 'FC', f'MI_allsujet.nc')):
            print(f'ALREADY DONE MI')
            return

    #### generate pairs
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:
        
        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    #### initiate res
    if stretch:
        time_vec = np.arange(stretch_point_ERP)
    else:
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

    MI_allsujet = np.zeros((len(sujet_list), len(pairs_to_compute), len(cond_list), time_vec.size))

    #### compute
    #sujet = sujet_list[0]
    # def get_MI_sujet(sujet, stretch):
    for sujet in sujet_list:

        print(sujet)

        #### identify anat info
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:
            
            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')

        #### compute
        cond_sel = ['VS', 'CHARGE']

        if stretch:
            time_vec = np.arange(stretch_point_ERP)
        else:
            time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

        #### LOAD DATA
        print('COMPUTE ERP')
        erp_data = {}

        cond_sel = ['VS', 'CHARGE']

        respfeatures = load_respfeatures(sujet)

        if stretch:

            #cond = 'VS'
            for cond in cond_sel:

                erp_data[cond] = {}

                data = load_data_sujet(sujet, cond)
                respfeatures_i = respfeatures[cond]
                inspi_starts = respfeatures_i['inspi_index'].values

                #chan, chan_i = A, chan_list_eeg_short.tolist().index(A)
                for chan in chan_list_eeg_short:

                    x = data[chan_list_eeg.tolist().index(chan),:]
                    x = scipy.signal.detrend(x, type='linear')
                    # data = zscore(data)
                    data_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_ERP, x, srate)

                    erp_data[cond][chan] = data_stretch

        else:
                
            #cond = 'VS'
            for cond in cond_sel:

                erp_data[cond] = {}

                data = load_data_sujet(sujet, cond)
                respfeatures_i = respfeatures[cond]
                inspi_starts = respfeatures_i['inspi_index'].values

                #chan, chan_i = A, chan_list_eeg_short.tolist().index(A)
                for chan in chan_list_eeg_short:

                    #### chunk
                    data_ERP = np.zeros((inspi_starts.shape[0], time_vec.size))

                    #### load
                    x = data[chan_list_eeg.tolist().index(chan),:]

                    #### low pass 45Hz + detrend
                    x = scipy.signal.detrend(x, type='linear')
                    x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)
                    x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                    for start_i, start_time in enumerate(inspi_starts):

                        t_start = int(start_time + ERP_time_vec[0]*srate)
                        t_stop = int(start_time + ERP_time_vec[-1]*srate)

                        if t_start < 0 or t_stop > x.size:
                            continue

                        x_chunk = x[t_start: t_stop]

                        data_ERP[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()

                    erp_data[cond][chan] = data_ERP

        #### compute MI
        print('COMPUTE MI')

        #pair_i, pair = 0, pairs_to_compute[0]
        for pair_i, pair in enumerate(pairs_to_compute):

            print_advancement(pair_i, len(pairs_to_compute), [25,50,75])

            A, B = pair.split('-')[0], pair.split('-')[1]
            chan_sel = {A : chan_list_eeg_short.tolist().index(A), B : chan_list_eeg_short.tolist().index(B)}
                    
            #cond_i, cond = 0, 'VS'
            for cond_i, cond in enumerate(cond_sel):

                A_data = erp_data[cond][A]
                B_data = erp_data[cond][B]

                for i in range(time_vec.size):

                    MI_allsujet[sujet_list.index(sujet), pair_i, cond_i, i] = get_MI_2sig(A_data[:,i], B_data[:,i])

            if debug:

                plt.plot(MI_allsujet[0,0,0,:])
                plt.show()

                fig, ax = plt.subplots()

                for cond_i, cond in enumerate(cond_sel):

                    ax.plot(MI_allsujet[0,pair_i,cond_i, :], label=cond)

                plt.legend()
                plt.suptitle(sujet)
                plt.show()

    #### parallel
    # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_MI_sujet)(sujet, stretch_token) for sujet, stretch_token in zip(sujet_list, [stretch]*len(sujet_list)))

    #### export
    MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_list, 'time' : time_vec}

    xr_MI = xr.DataArray(data=MI_allsujet, dims=MI_dict.keys(), coords=MI_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

    if stretch:
        xr_MI.to_netcdf(f'MI_allsujet_stretch.nc')
    else:
        xr_MI.to_netcdf(f'MI_allsujet.nc')


    if debug:

        pairs_to_compute
        plt.plot(xr_MI.loc[:,'C4-Cz','VS',:].mean('sujet').values, label='VS')
        plt.plot(xr_MI.loc[:,'C4-Cz','CHARGE',:].mean('sujet').values, label='CHARGE')
        plt.legend()
        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        plt.savefig('test')












################################
######## WPLI ISPC ######## 
################################


def compilation_ispc_wpli(stretch):

    #### verify computation
    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'ISPC', f'ISPC_allsujet_stretch.nc')) and os.path.exists(os.path.join(path_precompute, 'FC', 'WPLI', f'WPLI_allsujet_stretch.nc')):
            print(f'ALREADY DONE')
            return

    else:

        if os.path.exists(os.path.join(path_precompute, 'FC', 'ISPC', f'ISPC_allsujet.nc')) and os.path.exists(os.path.join(path_precompute, 'FC', 'WPLI', f'WPLI_allsujet.nc')):
            print(f'ALREADY DONE')
            return

    #### params
    if stretch:
        time_vec = np.arange(stretch_point_ERP)
        
    else:
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')
    
    #### prep compute
    # xr_data = np.zeros((len(sujet_list), len(freq_band_fc_list), len(cond_list), 2, len(pairs_to_compute), time_vec.shape[0]))
    os.chdir(path_memmap)
    xr_data_ispc = np.memmap(f'res_fc_ispc_{stretch}.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(freq_band_fc_list), len(cond_list), len(pairs_to_compute), time_vec.shape[0]))
    xr_data_wpli = np.memmap(f'res_fc_wpli_{stretch}.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(freq_band_fc_list), len(cond_list), len(pairs_to_compute), time_vec.shape[0]))
    xr_dict = {'sujet':sujet_list, 'band':freq_band_fc_list, 'cond':cond_list, 'pair':pairs_to_compute, 'time':time_vec}

    params_list = []

    for sujet_i, sujet in enumerate(sujet_list):

        for cond_i, cond in enumerate(cond_list):

            for band_i, band in enumerate(freq_band_fc_list):

                params_list.append([stretch, sujet, cond, band])

                # res_fc_phase = get_pli_ispc(stretch, sujet, cond, band)
                # xr_data[sujet_i, band_i, cond_i, 0,:,:,:], xr_data[sujet_i, band_i, cond_i, 1,:,:,:] = res_fc_phase[0,:,:,:], res_fc_phase[1,:,:,:]

    ######## COMPUTE FUNCTION ########
    #sujet, cond, band = sujet_list[0], cond_list[0], freq_band_fc_list[0]
    def get_pli_ispc(stretch, sujet, cond, band):

        print(f'{sujet} {cond} {band} stretch:{stretch}')

        #### load data
        data = load_data_sujet(sujet, cond)
        data = data[[chan_i for chan_i, chan in enumerate(chan_list_eeg) if chan in chan_list_eeg_short]]
        
        data_length = data.shape[-1]

        wavelets = get_wavelets_fc(freq_band_fc[band])

        respfeatures_allcond = load_respfeatures(sujet)

        #### initiate res
        convolutions = np.zeros((len(chan_list_eeg_short), wavelets.shape[0], data_length), dtype=np.complex128)

        print('CONV')

        #nchan_i = 0
        # def convolution_x_wavelets_nchan(nchan_i, nchan):
        for nchan_i in range(chan_list_eeg_short.size):

            print_advancement(nchan_i, len(chan_list_eeg_short), steps=[25, 50, 75])
            
            nchan_conv = np.zeros((wavelets.shape[0], np.size(data,1)), dtype='complex')

            x = data[nchan_i,:]

            for fi in range(wavelets.shape[0]):

                nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

            convolutions[nchan_i,:,:] = nchan_conv

        # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(chan_list_eeg))    

        #### verif conv
        if debug:
            plt.plot(convolutions[0,0,:])
            plt.show()

        #### compute index
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')

        ######## FC / DFC ########
        if stretch:
            time_vec = np.arange(stretch_point_ERP)
            
        else:
            time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
        
        # res_fc_phase = np.zeros((2, len(pairs_to_compute), time_vec.shape[0]))

        print('COMPUTE FC')

        #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
        # def compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute):
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):

            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

            pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
            pair_A_i, pair_B_i = np.where(chan_list_eeg_short == pair_A)[0][0], np.where(chan_list_eeg_short == pair_B)[0][0]

            as1 = convolutions[pair_A_i,:,:]
            as2 = convolutions[pair_B_i,:,:]

            if stretch:

                #### stretch data
                as1_chunk = stretch_data_tf(respfeatures_allcond[cond], stretch_point_ERP, as1, srate)[0]
                as2_chunk = stretch_data_tf(respfeatures_allcond[cond], stretch_point_ERP, as2, srate)[0]

            else:

                inspi_starts = respfeatures_allcond[cond]['inspi_index'].values

                as1_chunk = np.zeros((inspi_starts.size, wavelets.shape[0], time_vec.size), dtype=np.complex128)
                as2_chunk = np.zeros((inspi_starts.size, wavelets.shape[0], time_vec.size), dtype=np.complex128)

                remove_i_list = []

                for start_i, start_time in enumerate(inspi_starts):

                    t_start = int(start_time + ERP_time_vec[0]*srate)
                    t_stop = int(start_time + ERP_time_vec[-1]*srate)

                    if t_start < 0 or t_stop > x.size:
                        remove_i_list.append(start_i)
                        continue

                    as1_chunk[start_i,:,:] = as1[:,t_start:t_stop]
                    as2_chunk[start_i,:,:] = as2[:,t_start:t_stop]

                if len(remove_i_list) != 0:
                    as1_chunk = as1_chunk[[i for i in range(inspi_starts.size) if i not in remove_i_list]]
                    as2_chunk = as2_chunk[[i for i in range(inspi_starts.size) if i not in remove_i_list]]

            ##### collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1_chunk)-np.angle(as2_chunk)))
            
            ##### compute ISPC and WPLI (and average over trials!)
            ispc_freq = np.abs(np.mean(cdd, axis=0))
            # res_fc_phase[0, pair_to_compute_i, :] = np.mean(ispc_freq, axis=0) #mean along freq
            xr_data_ispc[sujet_list.index(sujet),freq_band_fc_list.index(band),cond_list.index(cond),pair_to_compute_i,:] = np.mean(ispc_freq, axis=0)

            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            wpli_freq = np.abs( np.mean( np.imag(cdd), axis=1 ) ) / np.mean( np.abs( np.imag(cdd) ), axis=1 )
            # res_fc_phase[1, pair_to_compute_i, :] = np.mean(wpli_freq, axis=0)
            xr_data_wpli[sujet_list.index(sujet),freq_band_fc_list.index(band),cond_list.index(cond),pair_to_compute_i,:] = np.mean(wpli_freq, axis=0)

            if debug:

                plt.pcolormesh(ispc_freq)
                plt.show()

                plt.pcolormesh(wpli_freq)
                plt.show()

        # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

        # return res_fc_phase


    ######## COMPUTE ########
    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_pli_ispc)(stretch, sujet, cond, band) for stretch, sujet, cond, band in params_list)

    #### save
    xr_ispc = xr.DataArray(data=xr_data_ispc, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli = xr.DataArray(data=xr_data_wpli, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'FC', 'ISPC'))
    if stretch:
        xr_ispc.to_netcdf(f"ISPC_allsujet_stretch.nc")
    else:
        xr_ispc.to_netcdf(f"ISPC_allsujet.nc")

    os.chdir(os.path.join(path_precompute, 'FC', 'WPLI'))
    if stretch:
        xr_wpli.to_netcdf(f"WPLI_allsujet_stretch.nc")
    else:
        xr_wpli.to_netcdf(f"WPLI_allsujet.nc")

    #### clean
    os.chdir(path_memmap)
    try:
        os.remove(f'res_fc_ispc_{stretch}.dat')
        del xr_data
    except:
        pass

    try:
        os.remove(f'res_fc_wpli_{stretch}.dat')
        del xr_data
    except:
        pass






################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    #stretch = True
    for stretch in [True, False]:

        #get_MI_allsujet(stretch)
        execute_function_in_slurm_bash('n07_precompute_FC', 'get_MI_allsujet', [stretch], n_core=15, mem='15G')
        #sync_folders__push_to_crnldata()

        #compilation_ispc_wpli(stretch)
        execute_function_in_slurm_bash('n07_precompute_FC', 'compilation_ispc_wpli', [stretch], n_core=15, mem='50G')
        #sync_folders__push_to_crnldata()
    


