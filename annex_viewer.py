



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False



################################
######## LOAD DATA ########
################################


#sujet, cond = sujet_list[14], 'VS'
def open_raw_data(sujet, cond):

    ######## identify project and sujet ########        
    sujet_project = sujet_project_nomenclature[sujet[2:4]]
    sujet_init_name = list(sujet_list_correspondance.keys())[list(sujet_list_correspondance.values()).index(sujet)][3:]

    ######## OPEN DATA ########
    if sujet_project == 'NORMATIVE':

        os.chdir(os.path.join(path_data, sujet_project, 'first', sujet_init_name))

        print(f"OPEN {sujet_project} : {sujet}")

        _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_{cond}_ValidICM.vhdr")
        _chan_list_eeg = _data.info['ch_names'][:-5]
        _data_eeg = _data.get_data()[:-5,:]
        pression_chan_i = _data.info['ch_names'].index('Pression')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig = _data.annotations.onset

        if debug:
            mne.viz.plot_raw(_data, n_channels=1)

            plt.plot(_respi)
            plt.show()

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    elif sujet_project == 'PHYSIOLOGY':

        os.chdir(os.path.join(path_data, sujet_project, sujet_init_name))

        print(f"OPEN {sujet_project} : {sujet}")

        _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_CONTINU_64Ch_A2Ref.vhdr")
        if sujet == '21PH_SB':
            _chan_list_eeg = _data.info['ch_names'][:-4]
            _data_eeg = _data.get_data()[:-4,:]
        else:
            _chan_list_eeg = _data.info['ch_names'][1:-4]
            _data_eeg = _data.get_data()[1:-4,:]
        pression_chan_i = _data.info['ch_names'].index('Pression')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig = _data.annotations.onset

        if debug:
            mne.viz.plot_raw(_data, n_channels=1)

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

        #### chunk cond
        start, stop = int(section_timming_PHYSIOLOGY[sujet][cond][0]*_srate_init), int(section_timming_PHYSIOLOGY[sujet][cond][1]*_srate_init)
        
        _data_eeg = _data_eeg[:,start:stop]
        _respi = _respi[start:stop]
        _trig = _trig[(_trig>=section_timming_PHYSIOLOGY[sujet][cond][0]) & (_trig<=section_timming_PHYSIOLOGY[sujet][cond][1])]
        _trig -= start

        if debug:

            plt.plot(_respi)
            plt.vlines(_trig*_srate_init, ymin=_respi.min(), ymax=_respi.max(), color='r')
            plt.show()

    elif sujet_project == 'ITL_LEO':

        os.chdir(os.path.join(path_data, 'ITL_LEO'))

        print(f"OPEN {sujet_project} : {sujet}")

        if cond == 'VS':
            cond_to_search = 'VS'
        elif cond == 'CHARGE':
            cond_to_search = 'ITL'

        file_name = [file for file in os.listdir() if file.find(sujet_init_name) != -1 and file.find(f'{cond_to_search}.edf') != -1][0]
        file_name_marker = [file for file in os.listdir() if file.find(sujet_init_name) != -1 and file.find(f'{cond_to_search}.Markers') != -1][0]

        _data = mne.io.read_raw_edf(file_name)
        _chan_list_eeg = _data.info['ch_names'][:-3]
        _data_eeg = _data.get_data()[:-3,:]
        pression_chan_i = _data.info['ch_names'].index('PRESSION')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        f = open(file_name_marker, "r")
        _trig = [int(line.split(',')[2][1:]) for line_i, line in enumerate(f.read().split('\n')) if len(line.split(',')) == 5 and line.split(',')[0] == 'Response']
        _trig = np.array(_trig) / _srate_init

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    elif sujet_project == 'DYSLEARN':

        print(f"OPEN {sujet_project} : {sujet}")

        if cond == 'CHARGE':
            cond_corrected = 'ITL'
        else:
            cond_corrected = cond

        os.chdir(os.path.join(path_data, sujet_project, cond_corrected))

        if sujet_init_name in ['08']:
            file_open = [file for file in os.listdir() if file.find('vhdr') != -1 and file.find(f"DYSLEARN_00{sujet_init_name}") != -1][-1]
        else:
            file_open = [file for file in os.listdir() if file.find('vhdr') != -1 and file.find(f"DYSLEARN_00{sujet_init_name}") != -1][0]    

        _data = mne.io.read_raw_brainvision(file_open)
        _chan_list_eeg = _data.info['ch_names'][:-3]
        _data_eeg = _data.get_data()[:-3,:]
        pression_chan_i = _data.info['ch_names'].index('PRESS')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig_onset = _data.annotations.onset
        _trig_name = _data.annotations.description

        if sujet == '42DL_11':
            start, stop = int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} DEBUT')[0][0]]*_srate_init), int(_trig_onset[np.where(_trig_name == f'Comment/VS FIN')[0][0]]*_srate_init)
        else:
            start, stop = int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} DEBUT')[0][0]]*_srate_init), int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} FIN')[0][0]]*_srate_init)
        
        _data_eeg = _data_eeg[:,start:stop]
        _respi = _respi[start:stop]

        _trig = _trig_onset[(_trig_onset>=start/_srate_init) & (_trig_onset<=stop/_srate_init)]
        _trig -= start/_srate_init

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    ######## ADJUST RESPI ########
    if sujet_respi_adjust[sujet] == 'inverse':
        _respi *= -1

    ######## RESAMPLE ########
    if _srate_init != 500:

        #### EEG
        _time_vec_origin = np.arange(0, _data_eeg.shape[-1]/_srate_init, 1/_srate_init)
        _time_vec_dwsampled = np.arange(0, _data_eeg.shape[-1]/_srate_init, 1/srate)

        _data_dwsampled = np.zeros((len(chan_list_eeg), _time_vec_dwsampled.shape[0]))

        for chan_i in range(_data_eeg.shape[0]):
            x = _data_eeg[chan_i,:]
            x_dwsampled = np.interp(_time_vec_dwsampled, _time_vec_origin, x)
            _data_dwsampled[chan_i,:] = x_dwsampled

        _data_eeg = _data_dwsampled

        #### AUX
        _respi = np.interp(_time_vec_dwsampled, _time_vec_origin, _respi) 

    ######## EXTRACT TRIG ########

    if debug:

        time_vec = np.arange(0, _respi.shape[0]/srate, 1/srate)
            
        plt.plot(time_vec, _respi)
        plt.title(f"{sujet}, {cond}")
        plt.vlines(_trig, ymin=_respi.min(), ymax=_respi.max(), color='r')
        plt.show()

    return _data_eeg, _respi, _trig










########################
######## VIEWER ########
########################



def viewer_one_sujet(sujet, cond, chan_selection, filter=False, raw_signals=False):

    #### params
    chan_list_i = [chan_i for chan_i, chan in enumerate(chan_list) if chan in chan_selection]
    chan_list_i.insert(0, np.where(chan_list == 'pression' )[0][0])

    #### load data
    print('load')
    if raw_signals:
        _data_eeg, _respi, _trig = open_raw_data(sujet, cond)
        data = np.concatenate((_data_eeg, _respi.reshape(1,-1)), axis=0)[chan_list_i,:]
    else:
        data = load_data_sujet(sujet, cond)[chan_list_i,:]

    chan_labels = ['respi']
    chan_labels.extend([f"{chan}" for chan_i, chan in enumerate(chan_selection)])

    if debug:

        plt.plot(data[0,:])
        plt.show()

        respfeatures = load_respfeatures(sujet)[cond]

        _x = zscore(data[-1,:])
        _respi = zscore(data[1,:])+5

        s = 100
        plt.plot(_respi, color='k', zorder=0)
        plt.plot(_x, zorder=0)
        plt.scatter(respfeatures['inspi_index'], _respi[respfeatures['inspi_index']], color='g', label='inspi', s=s, zorder=1)
        plt.scatter(respfeatures['expi_index'], _respi[respfeatures['expi_index']], color='b', label='expi', s=s, zorder=1)
        plt.scatter(respfeatures['inspi_index'], _x[respfeatures['inspi_index']], color='g', label='inspi', s=s, zorder=1)
        plt.scatter(respfeatures['expi_index'], _x[respfeatures['expi_index']], color='b', label='expi', s=s, zorder=1)
        plt.legend()
        plt.show()

    #### downsample
    print('resample')
    srate_downsample = 50

    time_vec = np.linspace(0,data.shape[-1],data.shape[-1])/srate
    time_vec_resample = np.linspace(0,data.shape[-1],int(data.shape[-1] * (srate_downsample / srate)))/srate

    data_resampled = np.zeros((data.shape[0], time_vec_resample.shape[0]))

    for chan_i in range(data.shape[0]):
        f = scipy.interpolate.interp1d(time_vec, data[chan_i,:], kind='quadratic', fill_value="extrapolate")
        data_resampled[chan_i,:] = f(time_vec_resample)

    if debug:

        plt.plot(time_vec, data[chan_i,:], label='raw')
        plt.plot(time_vec_resample, data_resampled[chan_i,:], label='resampled')
        plt.legend()
        plt.show()

    #### for one chan
    print('plot')
    if len(chan_selection) == 1:

        x = data_resampled[-1,:]

        respi = data_resampled[0,:]

        if filter:

            fcutoff = 40
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 0,0,1,1 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order+1,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)


            fcutoff = 100
            transw  = .2
            order   = np.round( 7*srate/fcutoff )
            shape   = [ 1,1,0,0 ]
            frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
            filtkern = scipy.signal.firls(order,frex,shape,fs=srate)
            x = scipy.signal.filtfilt(filtkern,1,x)

        chan_i = 1

        fig, ax = plt.subplots()
        
        ax.plot(time_vec_resample, zscore(respi)+3, label=chan_labels[0])
    
        ax.plot(time_vec_resample, zscore(x)+3*(chan_i+1), label=chan_labels[chan_i])

        ax.set_title(f"{sujet} {cond} raw:{raw_signals}")
        plt.legend()

        plt.show()

    #### for several chan
    else:

        chan_list_data_resampled = np.arange(data_resampled.shape[0])

        respi = data_resampled[1,:]
        ecg = data_resampled[0,:]

        fig, ax = plt.subplots()

        ax.plot(time_vec_resample, zscore(ecg), label=chan_labels[0])
        ax.plot(time_vec_resample, zscore(respi)+3, label=chan_labels[1])

        for chan_count, chan_i in enumerate(chan_list_data_resampled[2:]):
        
            x = data_resampled[chan_i,:]
            
            if filter:

                fcutoff = 40
                transw  = .2
                order   = np.round( 7*srate/fcutoff )
                shape   = [ 0,0,1,1 ]
                frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
                filtkern = scipy.signal.firls(order+1,frex,shape,fs=srate)
                x = scipy.signal.filtfilt(filtkern,1,x)


                fcutoff = 100
                transw  = .2
                order   = np.round( 7*srate/fcutoff )
                shape   = [ 1,1,0,0 ]
                frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]
                filtkern = scipy.signal.firls(order,frex,shape,fs=srate)
                x = scipy.signal.filtfilt(filtkern,1,x)

            ax.plot(time_vec_resample, zscore(x)+3*(chan_count+2), label=chan_labels[chan_i])
        
        ax.set_title(f"{sujet} {cond} raw:{raw_signals}")
        plt.legend()

        plt.show()















################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### sujet
    sujet_list = ['01NM_MW', '02NM_OL', '03NM_MC', '04NM_LS', '05NM_JS', '06NM_HC', '07NM_YB', '08NM_CM', '09NM_CV', '10NM_VA', '11NM_LC', '12NM_PS', '13NM_JP', '14NM_LD',
              '15PH_JS',  '16PH_LP',  '17PH_SB',  '18PH_TH',  '19PH_VA',  '20PH_VS',
              '21IL_NM', '22IL_DG', '23IL_DM', '24IL_DJ', '25IL_DC', '26IL_AP', '27IL_SL', '28IL_LL', '29IL_VR', '30IL_LC', '31IL_MA', '32IL_LY', '33IL_BA', '34IL_CM', '35IL_EA', '36IL_LT',
              '37DL_05', '38DL_06', '39DL_07', '40DL_08', '41DL_11', '42DL_12', '43DL_13', '44DL_14', '45DL_15', '46DL_16', '47DL_17', '48DL_18', '49DL_19', '50DL_20', '51DL_21', '52DL_22',
              '53DL_23', '54DL_24', '55DL_25', '56DL_26', '57DL_27', '58DL_28', '59DL_29', '60DL_30', '61DL_31', '62DL_32', '63DL_34',
              ]

    sujet = '02NM_OL'

    #### cond    
    cond = 'VS'
    cond = 'CHARGE'


    #### chan
    chan_list = ['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']

    chan_selection = ['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8']

    chan_selection = ['C3', 'C4', 'CP1', 'CP2', 'Cz', 'F3', 'F4', 'FC1', 'FC2', 'Fz']

    chan_selection = ['C3']

    #### view
    filter = False
    raw_signals = True
    raw_signals = False

    viewer_one_sujet(sujet, cond, chan_selection, filter=filter, raw_signals=raw_signals)


