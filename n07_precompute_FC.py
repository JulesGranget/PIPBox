

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False





################################
######## NO STRETCH ########
################################


#stretch = True
# def get_MI_allsujet_nostretch():

#     #### verify computation
#     if os.path.exists(os.path.join(path_precompute, 'FC', f'MI_allsujet.nc')):
#         print(f'ALREADY DONE MI')
#         return
        
#     #### params
#     cond_sel = ['VS', 'CHARGE']

#     #### generate pairs
#     pairs_to_compute = []

#     for pair_A in chan_list_eeg_short:
        
#         for pair_B in chan_list_eeg_short:

#             if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
#                 continue

#             pairs_to_compute.append(f'{pair_A}-{pair_B}')

#     #### initiate res
#     time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

#     MI_allsujet = np.zeros((len(sujet_list), len(pairs_to_compute), len(cond_list), time_vec.size))

#     #### compute
#     #sujet = sujet_list[0]
#     # def get_MI_sujet(sujet, stretch):
#     for sujet in sujet_list:

#         print(sujet)        

#         #### LOAD DATA
#         print('COMPUTE ERP')
#         erp_data = {}

#         respfeatures = load_respfeatures(sujet)
                
#         #cond = 'VS'
#         for cond in cond_sel:

#             erp_data[cond] = {}

#             data = load_data_sujet(sujet, cond)
#             respfeatures_i = respfeatures[cond]
#             inspi_starts = respfeatures_i['inspi_index'].values

#             #chan, chan_i = A, chan_list_eeg_short.tolist().index(A)
#             for chan in chan_list_eeg_short:

#                 #### chunk
#                 data_ERP = np.zeros((inspi_starts.shape[0], time_vec.size))

#                 #### load
#                 x = data[chan_list_eeg.tolist().index(chan),:]

#                 #### low pass 45Hz + detrend
#                 x = scipy.signal.detrend(x, type='linear')
#                 x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)
#                 x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

#                 for start_i, start_time in enumerate(inspi_starts):

#                     t_start = int(start_time + ERP_time_vec[0]*srate)
#                     t_stop = int(start_time + ERP_time_vec[-1]*srate)

#                     if t_start < 0 or t_stop > x.size:
#                         continue

#                     x_chunk = x[t_start: t_stop]

#                     data_ERP[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()

#                 erp_data[cond][chan] = data_ERP

#         #### compute MI
#         print('COMPUTE MI')

#         #pair_i, pair = 0, pairs_to_compute[0]
#         for pair_i, pair in enumerate(pairs_to_compute):

#             print_advancement(pair_i, len(pairs_to_compute), [25,50,75])

#             A, B = pair.split('-')[0], pair.split('-')[1]
                    
#             #cond_i, cond = 0, 'VS'
#             for cond_i, cond in enumerate(cond_sel):

#                 A_data = erp_data[cond][A]
#                 B_data = erp_data[cond][B]

#                 for i in range(time_vec.size):

#                     MI_allsujet[sujet_list.index(sujet), pair_i, cond_i, i] = get_MI_2sig(A_data[:,i], B_data[:,i])

#             if debug:

#                 plt.plot(MI_allsujet[0,0,0,:])
#                 plt.show()

#                 fig, ax = plt.subplots()

#                 for cond_i, cond in enumerate(cond_sel):

#                     ax.plot(MI_allsujet[0,pair_i,cond_i, :], label=cond)

#                 plt.legend()
#                 plt.suptitle(sujet)
#                 plt.show()

#     #### parallel
#     # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_MI_sujet)(sujet, stretch_token) for sujet, stretch_token in zip(sujet_list, [stretch]*len(sujet_list)))

#     #### export
#     MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_list, 'time' : time_vec}

#     xr_MI = xr.DataArray(data=MI_allsujet, dims=MI_dict.keys(), coords=MI_dict.values())
    
#     os.chdir(os.path.join(path_precompute, 'FC', 'MI'))

#     xr_MI.to_netcdf(f'MI_allsujet.nc')


#     if debug:

#         pairs_to_compute
#         plt.plot(xr_MI.loc[:,'C4-Cz','VS',:].mean('sujet').values, label='VS')
#         plt.plot(xr_MI.loc[:,'C4-Cz','CHARGE',:].mean('sujet').values, label='CHARGE')
#         plt.legend()
#         os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
#         plt.savefig('test')








########################################
######## MUTUAL INFORMATION ########
########################################


def get_MI_sujet_stretch(sujet):

    #### verify computation
    if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_{sujet}_stretch.nc')):
        print(f'ALREADY DONE MI STRETCH')
        return
    
    #### params
    os.chdir(path_prep)
    xr_norm_params = xr.open_dataarray('norm_params.nc')
    respfeatures = load_respfeatures(sujet)

    cond_sel = ['VS', 'CHARGE']

    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    #### container
    MI_sujet = np.zeros((len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

    #cond_i, cond = 0, 'VS'
    for cond_i, cond in enumerate(cond_sel):

        print(cond)

        data = load_data_sujet_CSD(sujet, cond)
        data_rscore = (data - xr_norm_params.loc[sujet, 'CSD', 'median', :].values.reshape(-1,1)) * 0.6745 / xr_norm_params.loc[sujet, 'CSD', 'mad', :].values.reshape(-1,1)
        
        if debug:

            plt.plot(scipy.stats.zscore(data[0,:]), label='raw')
            plt.plot(scipy.stats.zscore(data_rscore[0,:]), label='rscore')
            plt.legend()
            plt.show()

        respfeatures_i = respfeatures[cond]

        #pair_i, pair = 0, pairs_to_compute[0]
        # for pair_i, pair in enumerate(pairs_to_compute):
        def get_MI_pair(pair_i, pair):

            print_advancement(pair_i, len(pairs_to_compute), [25,50,75])

            A, B = pair.split('-')[0], pair.split('-')[1]

            x = data[chan_list_eeg.tolist().index(A),:]
            y = data[chan_list_eeg.tolist().index(B),:]

            #### pad sig for the conv
            x_pad = np.pad(x, int(MI_window_size/2), mode='reflect')
            y_pad = np.pad(y, int(MI_window_size/2), mode='reflect')

            MI_conv_pair = np.array([get_MI_2sig(x_pad[i:i+MI_window_size], y_pad[i:i+MI_window_size]) for i in range(int(x_pad.size-MI_window_size))])

            # MI_conv_pair = np.random.rand(x.size)

            if debug:

                plt.plot(MI_conv_pair)
                plt.show()

            #### stretch
            MI_stretch_pair, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_FC, MI_conv_pair, srate)

            #### cap epochs
            MI_stretch_pair = MI_stretch_pair[:nrespcycle_FC,:]

            if debug:

                for cycle_i in range(MI_stretch_pair.shape[0]):

                    plt.plot(MI_stretch_pair[cycle_i,:], alpha=0.2)

                plt.plot(np.median(MI_stretch_pair, axis=0), color='r')
                plt.show()

            return MI_stretch_pair

        #### parallelize
        res_pair_MI = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_MI_pair)(pair_i, pair) for pair_i, pair in enumerate(pairs_to_compute))  
        
        #### unpack
        for pair_i, pair in enumerate(pairs_to_compute):
            MI_sujet[cond_i, pair_i, :, :] = res_pair_MI[pair_i]

    #### export
    MI_dict = {'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}

    xr_MI_sujet = xr.DataArray(data=MI_sujet, dims=MI_dict.keys(), coords=MI_dict.values())
    
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    xr_MI_sujet.to_netcdf(f'MI_stretch_{sujet}.nc')

    print('done')












################################
######## WPLI ISPC ######## 
################################


# def get_ISPC_WPLI_nostretch():

#     #### verify computation
#     if os.path.exists(os.path.join(path_precompute, 'FC', 'ISPC', f'ISPC_allsujet.nc')) and os.path.exists(os.path.join(path_precompute, 'FC', 'WPLI', f'WPLI_allsujet.nc')):
#         print(f'ALREADY DONE')
#         return

#     #### params
#     time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

#     pairs_to_compute = []

#     for pair_A in chan_list_eeg_short:

#         for pair_B in chan_list_eeg_short:

#             if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
#                 continue

#             pairs_to_compute.append(f'{pair_A}-{pair_B}')
    
#     #### prep compute
#     # xr_data = np.zeros((len(sujet_list), len(freq_band_fc_list), len(cond_list), 2, len(pairs_to_compute), time_vec.shape[0]))
#     os.chdir(path_memmap)
#     xr_data_ispc = np.memmap(f'res_fc_ispc.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(freq_band_fc_list), len(cond_list), len(pairs_to_compute), time_vec.shape[0]))
#     xr_data_wpli = np.memmap(f'res_fc_wpli.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(freq_band_fc_list), len(cond_list), len(pairs_to_compute), time_vec.shape[0]))
#     xr_dict = {'sujet':sujet_list, 'band':freq_band_fc_list, 'cond':cond_list, 'pair':pairs_to_compute, 'time':time_vec}

#     params_list = []

#     for sujet_i, sujet in enumerate(sujet_list):

#         for cond_i, cond in enumerate(cond_list):

#             for band_i, band in enumerate(freq_band_fc_list):

#                 params_list.append([sujet, cond, band])

#                 # res_fc_phase = get_pli_ispc(stretch, sujet, cond, band)
#                 # xr_data[sujet_i, band_i, cond_i, 0,:,:,:], xr_data[sujet_i, band_i, cond_i, 1,:,:,:] = res_fc_phase[0,:,:,:], res_fc_phase[1,:,:,:]

#     ######## COMPUTE FUNCTION ########
#     #sujet, cond, band = sujet_list[0], cond_list[0], freq_band_fc_list[0]
#     def get_pli_ispc(stretch, sujet, cond, band):

#         print(f'{sujet} {cond} {band} stretch:{stretch}')

#         #### load data
#         data = load_data_sujet(sujet, cond)
#         data = data[[chan_i for chan_i, chan in enumerate(chan_list_eeg) if chan in chan_list_eeg_short]]
        
#         data_length = data.shape[-1]

#         wavelets = get_wavelets_fc(freq_band_fc[band])

#         respfeatures_allcond = load_respfeatures(sujet)

#         #### initiate res
#         convolutions = np.zeros((len(chan_list_eeg_short), wavelets.shape[0], data_length), dtype=np.complex128)

#         print('CONV')

#         #nchan_i = 0
#         # def convolution_x_wavelets_nchan(nchan_i, nchan):
#         for nchan_i in range(chan_list_eeg_short.size):

#             print_advancement(nchan_i, len(chan_list_eeg_short), steps=[25, 50, 75])
            
#             nchan_conv = np.zeros((wavelets.shape[0], np.size(data,1)), dtype='complex')

#             x = data[nchan_i,:]

#             for fi in range(wavelets.shape[0]):

#                 nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

#             convolutions[nchan_i,:,:] = nchan_conv

#         # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(chan_list_eeg))    

#         #### verif conv
#         if debug:
#             plt.plot(convolutions[0,0,:])
#             plt.show()

#         #### compute index
#         pairs_to_compute = []

#         for pair_A in chan_list_eeg_short:

#             for pair_B in chan_list_eeg_short:

#                 if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
#                     continue

#                 pairs_to_compute.append(f'{pair_A}-{pair_B}')

#         ######## FC / DFC ########
#         time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
        
#         # res_fc_phase = np.zeros((2, len(pairs_to_compute), time_vec.shape[0]))

#         print('COMPUTE FC')

#         #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
#         # def compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute):
#         for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):

#             print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

#             pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
#             pair_A_i, pair_B_i = np.where(chan_list_eeg_short == pair_A)[0][0], np.where(chan_list_eeg_short == pair_B)[0][0]

#             as1 = convolutions[pair_A_i,:,:]
#             as2 = convolutions[pair_B_i,:,:]

#             cross_corr = np.zeros((as1.shape), dtype='complex128')

#             for fi in range(wavelets.shape[0]):

#                 cross_corr[fi,:] = scipy.signal.correlate(as1[fi,:], as2[fi,:], mode='same', method='fft') # scipy.signal.csd() ?

#             inspi_starts = respfeatures_allcond[cond]['inspi_index'].values

#             as1_chunk = np.zeros((inspi_starts.size, wavelets.shape[0], time_vec.size), dtype=np.complex128)
#             as2_chunk = np.zeros((inspi_starts.size, wavelets.shape[0], time_vec.size), dtype=np.complex128)

#             as_chunk_crosscorr = np.zeros((inspi_starts.size, wavelets.shape[0], time_vec.size), dtype=np.complex128)

#             if debug:

#                 plt.pcolormesh(np.real(as_chunk_crosscorr.mean(axis=0)))
#                 plt.show()

#                 plt.plot(np.real(as_chunk_crosscorr.mean(axis=0).mean(axis=0)), label='crosscorr')
#                 plt.plot(np.real(as1_chunk.mean(axis=0).mean(axis=0)))
#                 plt.plot(np.real(as2_chunk.mean(axis=0).mean(axis=0)))
#                 plt.legend()
#                 plt.show()

#             remove_i_list = []

#             for start_i, start_time in enumerate(inspi_starts):

#                 t_start = int(start_time + ERP_time_vec[0]*srate)
#                 t_stop = int(start_time + ERP_time_vec[-1]*srate)

#                 if t_start < 0 or t_stop > x.size:
#                     remove_i_list.append(start_i)
#                     continue

#                 as1_chunk[start_i,:,:] = as1[:,t_start:t_stop]
#                 as2_chunk[start_i,:,:] = as2[:,t_start:t_stop]

#                 as_chunk_crosscorr[start_i,:,:] = cross_corr[:,t_start:t_stop]

#             if len(remove_i_list) != 0:
#                 as1_chunk = as1_chunk[[i for i in range(inspi_starts.size) if i not in remove_i_list]]
#                 as2_chunk = as2_chunk[[i for i in range(inspi_starts.size) if i not in remove_i_list]]

#                 as_chunk_crosscorr = as_chunk_crosscorr[[i for i in range(inspi_starts.size) if i not in remove_i_list]]

#             ##### collect "eulerized" phase angle differences
#             cdd = np.exp(1j*(np.angle(as1_chunk)-np.angle(as2_chunk)))
            
#             ##### compute ISPC and WPLI (and average over trials!)
#             ispc_freq = np.abs(np.mean(cdd, axis=0))
#             # res_fc_phase[0, pair_to_compute_i, :] = np.mean(ispc_freq, axis=0) #mean along freq
#             xr_data_ispc[sujet_list.index(sujet),freq_band_fc_list.index(band),cond_list.index(cond),pair_to_compute_i,:] = np.mean(ispc_freq, axis=0)

#             # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
#             wpli_freq = np.abs( np.mean( np.imag(as_chunk_crosscorr), axis=0 ) ) / np.mean( np.abs( np.imag(as_chunk_crosscorr) ), axis=0 )
#             # res_fc_phase[1, pair_to_compute_i, :] = np.mean(wpli_freq, axis=0)
#             xr_data_wpli[sujet_list.index(sujet),freq_band_fc_list.index(band),cond_list.index(cond),pair_to_compute_i,:] = np.mean(wpli_freq, axis=0)

#             if debug:

#                 plt.pcolormesh(ispc_freq)
#                 plt.show()

#                 plt.pcolormesh(wpli_freq)
#                 plt.show()

#                 plt.plot(np.mean(wpli_freq, axis=0), label='wpli')
#                 plt.plot(np.mean(ispc_freq, axis=0), label='ispc')
#                 plt.legend()
#                 plt.show()

#         # joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

#         # return res_fc_phase


#     ######## COMPUTE ########
#     joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_pli_ispc)(stretch, sujet, cond, band) for stretch, sujet, cond, band in params_list)

#     #### save
#     xr_ispc = xr.DataArray(data=xr_data_ispc, dims=xr_dict.keys(), coords=xr_dict.values())
#     xr_wpli = xr.DataArray(data=xr_data_wpli, dims=xr_dict.keys(), coords=xr_dict.values())

#     os.chdir(os.path.join(path_precompute, 'FC', 'ISPC'))
#     xr_ispc.to_netcdf(f"ISPC_allsujet.nc")

#     os.chdir(os.path.join(path_precompute, 'FC', 'WPLI'))
#     xr_wpli.to_netcdf(f"WPLI_allsujet.nc")



#sujet = sujet_list_FC[0]
def get_ISPC_WPLI_stretch(sujet):
    
    # Check if results already exist
    ispc_path = os.path.join(path_precompute, 'FC', 'ISPC', f'ISPC_{sujet}_stretch.nc')
    wpli_path = os.path.join(path_precompute, 'FC', 'WPLI', f'WPLI_{sujet}_stretch.nc')
    
    if os.path.exists(ispc_path) and os.path.exists(wpli_path):
        print(f'ALREADY DONE')
        return
    
    cond_sel = ['VS', 'CHARGE']
    
    pairs_to_compute = [f'{chan_list_eeg_short[i]}-{chan_list_eeg_short[j]}'
                        for i in range(len(chan_list_eeg_short))
                        for j in range(i + 1, len(chan_list_eeg_short))]
    
    def compute_pair(pair_i):

        os.chdir(path_prep)
        xr_norm_params = xr.open_dataarray('norm_params.nc')
        respfeatures_sujet = load_respfeatures(sujet)
    
        pair = pairs_to_compute[pair_i]
        pair_A, pair_B = pair.split('-')
        idx_A, idx_B = np.where(chan_list_eeg_short == pair_A)[0][0], np.where(chan_list_eeg_short == pair_B)[0][0]

        print(f"{pair} {int(pair_i*100/len(pairs_to_compute))}%", flush=True)
        
        ISPC_all_bands = np.zeros((len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
        WPLI_all_bands = np.zeros((len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
        
        for band_i, band in enumerate(freq_band_fc_list):

            wavelets = get_wavelets_fc(freq_band_fc[band])
            win_conv = ISPC_window_size[band]
            
            for cond_i, cond in enumerate(cond_sel):

                respfeatures_sujet_chunk = respfeatures_sujet[cond][:nrespcycle_FC+20] 
                len_sig_to_analyze = respfeatures_sujet_chunk['next_inspi_index'].values[-1]

                data = load_data_sujet_CSD(sujet, cond)[:,:len_sig_to_analyze]
                data = data[[idx_A, idx_B],:]
                data_rscore = (data - xr_norm_params.loc[sujet, 'CSD', 'median', [pair_A, pair_B]].values.reshape(-1,1)) \
                              * 0.6745 / xr_norm_params.loc[sujet, 'CSD', 'mad', [pair_A, pair_B]].values.reshape(-1,1)
                                
                convolutions = np.array([
                    np.array([scipy.signal.fftconvolve(data_rscore[ch, :], wavelet, 'same') for wavelet in wavelets])
                    for ch in [0, 1]
                ])
                
                x_conv, y_conv = convolutions[0], convolutions[1]

                x_pad, y_pad = np.pad(x_conv, ((0, 0), (win_conv//2, win_conv//2)), mode='reflect'), np.pad(y_conv, ((0, 0), (win_conv//2, win_conv//2)), mode='reflect')

                ISPC_conv = np.array([
                    [get_ISPC_2sig(x_pad[w, i:i+win_conv], y_pad[w, i:i+win_conv]) for i in range(data_rscore.shape[1])]
                    for w in range(x_conv.shape[0])
                ])

                WPLI_conv = np.array([
                    [get_WPLI_2sig(x_pad[w, i:i+win_conv], y_pad[w, i:i+win_conv]) for i in range(data_rscore.shape[1])]
                    for w in range(x_conv.shape[0])
                ])

                # ISPC_conv = np.random.random(x_conv.shape)
                # WPLI_conv = np.random.random(x_conv.shape)

                ISPC_conv = np.median(ISPC_conv, axis=0)
                WPLI_conv = np.median(WPLI_conv, axis=0)
                
                ISPC_stretch = stretch_data(respfeatures_sujet_chunk, stretch_point_FC, ISPC_conv, srate)[0]
                WPLI_stretch = stretch_data(respfeatures_sujet_chunk, stretch_point_FC, WPLI_conv, srate)[0]
                
                ISPC_all_bands[cond_i, band_i] = ISPC_stretch[:nrespcycle_FC,:]
                WPLI_all_bands[cond_i, band_i] = WPLI_stretch[:nrespcycle_FC,:]
                
        return ISPC_all_bands, WPLI_all_bands
    
    results = joblib.Parallel(n_jobs=n_core, prefer='processes', batch_size=1)(
        joblib.delayed(compute_pair)(pair_i) for pair_i in range(len(pairs_to_compute))
    )

    # Preallocate results
    ISPC_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
    WPLI_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
    
    for pair_i in range(len(pairs_to_compute)):
        ISPC_sujet[pair_i] = results[pair_i][0]
        WPLI_sujet[pair_i] = results[pair_i][1]
    
    xr_dict = {'pair': pairs_to_compute, 'cond': cond_sel, 'band': freq_band_fc_list, 'cycle': np.arange(nrespcycle_FC), 'time': np.arange(stretch_point_FC)}
    xr_ispc = xr.DataArray(data=ISPC_sujet, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli = xr.DataArray(data=WPLI_sujet, dims=xr_dict.keys(), coords=xr_dict.values())
        
    os.chdir(os.path.join(path_precompute, 'FC', 'ISPC'))
    xr_ispc.to_netcdf(f'ISPC_{sujet}_stretch.nc')
    os.chdir(os.path.join(path_precompute, 'FC', 'WPLI'))
    xr_wpli.to_netcdf(f'WPLI_{sujet}_stretch.nc')
    
    print('done')

    

    









################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET ########

    #get_MI_sujet_stretch()
    execute_function_in_slurm_bash('n07_precompute_FC', 'get_MI_sujet_stretch', [[sujet] for sujet in sujet_list_FC], n_core=15, mem='20G')
    #sync_folders__push_to_crnldata()

    #get_ISPC_WPLI_stretch()
    execute_function_in_slurm_bash('n07_precompute_FC', 'get_ISPC_WPLI_stretch', [[sujet] for sujet in sujet_list_FC], n_core=20, mem='30G')
    #sync_folders__push_to_crnldata()
    
