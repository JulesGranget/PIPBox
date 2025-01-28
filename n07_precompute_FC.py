

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



def get_MI_allsujet():

    #stretch = False
    for stretch in [True, False]:

        #### verify computation
        if stretch:

            if os.path.exists(os.path.join(path_precompute, 'FC', f'allsujet_MI_stretch.nc')):
                print(f'ALREADY DONE MI STRETCH')
                continue

        else:

            if os.path.exists(os.path.join(path_precompute, 'FC', f'allsujet_MI.nc')):
                print(f'ALREADY DONE MI')
                continue

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

        os.chdir(path_memmap)
        MI_allsujet = np.memmap(f'allsujet_MI_computation.dat', dtype=np.float32, mode='w+', shape=(len(sujet_list), len(pairs_to_compute), len(cond_list), time_vec.size))

        #### compute
        def get_MI_sujet(sujet, stretch=False):

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
                        x = scipy.signal.detrend(data, type='linear')
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
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_MI_sujet)(sujet, stretch_token) for sujet, stretch_token in zip(sujet_list, [stretch]*len(sujet_list)))

        #### export
        if stretch:
            MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_list, 'time' : time_vec}
        else:
            MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_list, 'phase' : time_vec}

        xr_MI = xr.DataArray(data=MI_allsujet, dims=MI_dict.keys(), coords=MI_dict.values())
        
        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

        if stretch:
            xr_MI.to_netcdf(f'allsujet_MI_allpairs_stretch.nc')
        else:
            xr_MI.to_netcdf(f'allsujet_MI_allpairs.nc')


        if debug:

            pairs_to_compute
            plt.plot(xr_MI.loc[:,'C4-Cz','VS',:].mean('sujet').values, label='VS')
            plt.plot(xr_MI.loc[:,'C4-Cz','CHARGE',:].mean('sujet').values, label='CHARGE')
            plt.legend()
            os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
            plt.savefig('test')

        os.chdir(path_memmap)
        try:
            os.remove(f'allsujet_MI_computation.dat.dat')
            del MI_allsujet
        except:
            pass










################################
######## PLI ISPC ######## 
################################

#sujet, cond, band = sujet_list[0], cond_list[0], freq_band_fc_list[0]
def get_pli_ispc_fc_dfc_trial(sujet, cond, band):

    #### load data
    data = load_data_sujet(sujet, cond)
    data = data[[chan_i for chan_i, chan in enumerate(chan_list_eeg) if chan in chan_list_eeg_short]]
    
    data_length = data.shape[-1]

    wavelets = get_wavelets_fc(band_prep, freq)

    respfeatures_allcond = load_respfeatures(sujet)

    #### initiate res
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(chan_list_eeg), nfrex_fc, data_length))

    #### generate fake convolutions
    # convolutions = np.random.random(len(prms['chan_list_ieeg']) * nfrex_fc * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1]) * 1j
    # convolutions += np.random.random(len(prms['chan_list_ieeg']) * nfrex_fc * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1]) 

    # convolutions = np.zeros((len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1])) 

    print('CONV')

    #nchan = 0
    def convolution_x_wavelets_nchan(nchan_i, nchan):

        print_advancement(nchan_i, len(chan_list_eeg), steps=[25, 50, 75])
        
        nchan_conv = np.zeros((nfrex_fc, np.size(data,1)), dtype='complex')

        x = data[nchan_i,:]

        for fi in range(nfrex_fc):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan_i,:,:] = nchan_conv

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(chan_list_eeg))

    #### free memory
    del data        

    #### verif conv
    if debug:
        plt.plot(convolutions[0,0,:])
        plt.show()

    #### compute index
    pairs_to_compute = []

    for pair_A in chan_list_eeg:

        for pair_B in chan_list_eeg:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    ######## FC / DFC ########

    os.chdir(path_memmap)
    res_fc_phase = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_phase.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), len(phase_list), nfrex_fc))

    #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
    def compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute):

        print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

        pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
        pair_A_i, pair_B_i = chan_list_eeg.index(pair_A), chan_list_eeg.index(pair_B)

        as1 = convolutions[pair_A_i,:,:]
        as2 = convolutions[pair_B_i,:,:]

        #### stretch data
        as1_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_ERP, as1, srate)[0]
        as2_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_ERP, as2, srate)[0]

        #phase = 'whole'
        for phase_i, phase in enumerate(phase_list):

            #### chunk
            if phase == 'whole':
                as1_stretch_chunk =  np.transpose(as1_stretch, (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch, (1, 0, 2)).reshape((nfrex_fc, -1))

            if phase == 'inspi':
                as1_stretch_chunk =  np.transpose(as1_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)], (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)], (1, 0, 2)).reshape((nfrex_fc, -1))

            if phase == 'expi':
                as1_stretch_chunk =  np.transpose(as1_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):], (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):], (1, 0, 2)).reshape((nfrex_fc, -1))

            ##### collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1_stretch_chunk)-np.angle(as2_stretch_chunk)))
            
            ##### compute ISPC and WPLI (and average over trials!)
            res_fc_phase[0, pair_to_compute_i, phase_i, :] = np.abs(np.mean(cdd, axis=1))
            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            res_fc_phase[1, pair_to_compute_i, phase_i, :] = np.abs( np.mean( np.imag(cdd), axis=1 ) ) / np.mean( np.abs( np.imag(cdd) ), axis=1 )

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

    if debug:
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[10, 20, 50, 75])
            compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute)

    res_fc_phase_export = res_fc_phase.copy()

    #### remove memmap
    os.chdir(path_memmap)
    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_phase.dat')
        del convolutions
    except:
        pass

    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_convolutions.dat')
        del res_fc_phase
    except:
        pass

    return res_fc_phase_export








def get_wpli_ispc_fc_dfc(sujet, cond):

    #band_prep = 'wb'
    for band_prep in band_prep_list:
        #band, freq = 'theta', [4,8]
        for band, freq in freq_band_dict_FC[band_prep].items():

            #### verif computation
            if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{cond}_o_{band}_allpairs.nc')):
                print(f'ALREADY DONE FC {cond} {band}')
                return

            #### identify anat info
            pairs_to_compute = []

            for pair_A in chan_list_eeg:
                
                for pair_B in chan_list_eeg:

                    if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                        continue

                    pairs_to_compute.append(f'{pair_A}-{pair_B}')

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                #### for dfc computation
                mat_fc = get_pli_ispc_fc_dfc_trial(sujet, cond, odor_i, band_prep, band, freq)

                if debug:
                    plt.plot(mat_fc[1,0,0,:], label='whole')
                    plt.plot(mat_fc[1,0,1,:], label='inspi')
                    plt.plot(mat_fc[1,0,2,:], label='expi')
                    plt.legend()
                    plt.show()

                #### export
                os.chdir(os.path.join(path_precompute, sujet, 'FC'))
                dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute, 'phase' : ['whole', 'inspi', 'expi'], 'nfrex' : range(nfrex_fc)}
                xr_export = xr.DataArray(mat_fc, coords=dict_xr.values(), dims=dict_xr.keys())
                xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{cond}_{odor_i}_{band}_allpairs.nc')








################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    #get_MI_allsujet()
    execute_function_in_slurm_bash('n06_precompute_FC', 'get_MI_allsujet', [], n_core=15, mem='15G')
    







    export_df_MI()



    ######## OTHER FC METRICS ########

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print('######## PRECOMPUTE DFC ########') 
        #cond = 'FR_CV_1'
        for cond in conditions:

            # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, monopol)
            execute_function_in_slurm_bash_mem_choice('n7_precompute_FC', 'get_wpli_ispc_fc_dfc', [sujet, cond], '35G')


