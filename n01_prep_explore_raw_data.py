



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import xarray as xr
import json

from n00_config_params import *

from n00bis_config_analysis_functions import *

debug = False



################################
######## LOAD DATA ########
################################

def export_all_df_alldata():

    df_info_data = pd.DataFrame()

    #project = project_name_list_raw[3]
    for project in project_name_list_raw:

        ######## COVEM ITL ########

        if project == 'COVEM_ITL':

            os.chdir(os.path.join(path_data, project))

            files_name = os.listdir()
                    
            #file = files_name[0]
            for file_i, file in enumerate(files_name):

                print(f"OPEN {project} : {file}")
                
                with open(file, 'r') as file_to_open:
                    data = json.load(file_to_open)

                if debug:
                    data['header']

                _sujet = [_sujet for _sujet in sujet_list_project_wise[project] if file.find(_sujet) != -1][0]
                _data = np.array(data['recording']['channelData'])
                _srate = data['header']['sampRate']
                _chan_list = data['header']['acquisitionLocation']
                _ref, _ground = data['header']['referencesLocation'][0], data['header']['groundsLocation'][0]
                _lowpass = np.nan

                df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : ['CHARGE'], 'data_shape' : [f"{_data.shape[0]}/{_data.shape[-1]}"], 
                                                                    'length.min' : [_data.shape[-1]/_srate/60], 'srate' : [_srate], 'nchan' : [len(_chan_list)], 'chan_list' : [_chan_list], 
                                                                    'ref' : [_ref], 'ground' : [_ground], 'lowpass' : [_lowpass]})])
                    
        ######## NORMATIVE ########

        if project == 'NORMATIVE':
                    
            #_sujet_i, _sujet = 0, sujet_list_project_wise[project][0]
            for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                os.chdir(os.path.join(path_data, project, 'first', _sujet))

                print(f"OPEN {project} : {_sujet}")
                
                #cond = condition_list_project_wise[project][0]
                for cond in condition_list_project_wise[project]:

                    if os.path.exists(os.path.join(path_data, project, 'first', _sujet, f"{_sujet}_{cond}_ValidICM.vhdr")) == False:

                        df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : [cond], 'data_shape' : [np.nan], 
                                                                        'length.min' : [np.nan], 'srate' : [np.nan], 'nchan' : [np.nan], 'chan_list' : [np.nan], 
                                                                        'ref' : [np.nan], 'ground' : [np.nan], 'lowpass' : [np.nan]})])

                    else:

                        _data = mne.io.read_raw_brainvision(f"{_sujet}_{cond}_ValidICM.vhdr")
                        _data_extract = _data.get_data()
                        _srate = _data.info['sfreq']
                        _chan_list = _data.info['ch_names']
                        _lowpass = _data.info['lowpass']

                        if debug:
                            _data.info

                        df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : [cond], 'data_shape' : [f"{_data_extract.shape[0]}/{_data_extract.shape[-1]}"], 
                                                                            'length.min' : [_data_extract.shape[-1]/_srate/60], 'srate' : [_srate], 'nchan' : [len(_chan_list)], 'chan_list' : [_chan_list], 
                                                                            'ref' : [np.nan], 'ground' : [np.nan], 'lowpass' : [_lowpass]})])

        ######## PHYSIOLOGY ########        

        if project == 'PHYSIOLOGY':

            #_sujet_i, _sujet = 0, sujet_list_project_wise[project][0]
            for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                if _sujet in ['MC05', 'OL04']:
                    continue

                os.chdir(os.path.join(path_data, project, _sujet))

                print(f"OPEN {project} : {_sujet}")
                
                _data = mne.io.read_raw_brainvision(f"{_sujet}_CONTINU_64Ch_A2Ref.vhdr")
                _data_extract = _data.get_data()
                _srate = _data.info['sfreq']
                _chan_list = _data.info['ch_names']
                _lowpass = _data.info['lowpass']

                cond = 'CHARGE/VS'

                if debug:
                    _data.info

                df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : [cond], 'data_shape' : [f"{_data_extract.shape[0]}/{_data_extract.shape[-1]}"], 
                                                                    'length.min' : [_data_extract.shape[-1]/_srate/60], 'srate' : [_srate], 'nchan' : [len(_chan_list)], 'chan_list' : [_chan_list], 
                                                                    'ref' : [np.nan], 'ground' : [np.nan], 'lowpass' : [_lowpass]})])

        ######## SLP ########        

        if project == 'SLP':

            #_sujet_i, _sujet = 0, sujet_list_project_wise[project][0]
            for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                os.chdir(os.path.join(path_data, project, _sujet))

                print(f"OPEN {project} : {_sujet}")
                
                _data = mne.io.read_raw_brainvision(f"64Ch_SLP_{_sujet}_A2Ref.vhdr")
                _data_extract = _data.get_data()
                _srate = _data.info['sfreq']
                _chan_list = _data.info['ch_names']
                _lowpass = _data.info['lowpass']

                cond = 'CHARGE'

                if debug:
                    _data.info

                df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : [cond], 'data_shape' : [f"{_data_extract.shape[0]}/{_data_extract.shape[-1]}"], 
                                                                    'length.min' : [_data_extract.shape[-1]/_srate/60], 'srate' : [_srate], 'nchan' : [len(_chan_list)], 'chan_list' : [_chan_list], 
                                                                    'ref' : [np.nan], 'ground' : [np.nan], 'lowpass' : [_lowpass]})])


        ######## ITL LEO ########        

        if project == 'ITL_LEO':

            os.chdir(os.path.join(path_data, 'ITL_LEO'))

            #_sujet = sujet_list_project_wise[project][0]
            for _sujet in sujet_list_project_wise[project]:

                #cond = condition_list_project_wise[project][0]
                for cond in condition_list_project_wise[project]:

                    file_name = [file for file in os.listdir() if file.find(_sujet) != -1 and file.find(f'{cond}.edf') != -1][0]
                    file_name_marker = [file for file in os.listdir() if file.find(_sujet) != -1 and file.find(f'{cond}.Markers') != -1][0]
                    
                    _data = mne.io.read_raw_edf(file_name)
                    _data_extract = _data.get_data()
                    _srate = _data.info['sfreq']
                    _chan_list = _data.info['ch_names']
                    _lowpass = _data.info['lowpass']

                    f = open(file_name_marker, "r")

                    if debug:
                        _data.info

                    df_info_data = pd.concat([df_info_data, pd.DataFrame({'project' : [project], 'sujet' : [_sujet], 'cond' : [cond], 'data_shape' : [f"{_data_extract.shape[0]}/{_data_extract.shape[-1]}"], 
                                                                        'length.min' : [_data_extract.shape[-1]/_srate/60], 'srate' : [_srate], 'nchan' : [len(_chan_list)], 'chan_list' : [_chan_list], 
                                                                        'ref' : [np.nan], 'ground' : [np.nan], 'lowpass' : [_lowpass]})])


    ######## SAVE DF ALLDATA ########  
    os.chdir(path_data)
    df_info_data.to_excel('df_info_alldata.xlsx')




def explore_all_data():

    #project = project_name_list_raw[2]
    for project in project_name_list_raw:

        ######## COVEM ITL ########

        if project == 'COVEM_ITL':

            os.chdir(os.path.join(path_data, project))

            files_name = os.listdir()

            time_vec_extraction = np.arange(0, params_extraction_data[project]['time_cutoff']*60, 1/srate)

            #file = files_name[0]
            for file_i, file in enumerate(files_name):

                ######## Open json ########

                print(f"OPEN {project} : {file}")
                
                with open(file, 'r') as file_to_open:
                    data = json.load(file_to_open)

                if debug:
                    data['header']
                
                ######## Extract data ########

                _sujet = [_sujet for _sujet in sujet_list_project_wise[project] if file.find(_sujet) != -1][0]
                _data = np.array(data['recording']['channelData'])
                _srate = data['header']['sampRate']
                _chan_list = data['header']['acquisitionLocation']
                _chan_list = [chan for chan in _chan_list if chan not in ['PSM', 'ACC', 'EXT1']]
                _ref, _ground = data['header']['referencesLocation'][0], data['header']['groundsLocation'][0]

                _xr_data = xr.DataArray(data=_data.astype('float'), dims=['x', 'y'], coords=[np.arange(_data.shape[0]), np.arange(_data.shape[1])])
                _data = _xr_data.interpolate_na(dim="y", method="linear").values

                ######## Upsampled ########

                _time_vec_origin = np.arange(0, _data.shape[1]/_srate, 1/_srate)
                _time_vec_upsampled = np.arange(0, _data.shape[1]/_srate, 1/srate)

                _data_upsampled = np.zeros((len(_chan_list), _time_vec_upsampled.shape[0]))

                for chan_i in range(_data.shape[0]):
                    x = _data[chan_i,:]
                    x_upsampled = np.interp(_time_vec_upsampled, _time_vec_origin, x)
                    _data_upsampled[chan_i,:] = x_upsampled

                _data_upsampled = _data_upsampled[np.newaxis,:,:]

                ######## Vizu data ########

                if debug:
                    for chan_i in range(_data.shape[0]):
                        plt.plot(zscore(_data[chan_i,:].astype('float')) + chan_i, label=_chan_list[chan_i])
                    plt.legend()
                    plt.show()

                ######## Trim data ########

                _data_upsampled = _data_upsampled[:,:,:time_vec_extraction.shape[0]]

                ######## generate xr ########

                _xr_data = xr.DataArray(data=_data_upsampled, dims=['sujet', 'chan', 'time'], coords=[[_sujet], _chan_list, time_vec_extraction])

                if file_i == 0:
                    _xr_data_allsujet = _xr_data
                else:
                    _xr_data_allsujet = xr.concat([_xr_data_allsujet, _xr_data], dim='sujet')

            ######## inspect data ########
            if debug:
                for sujet in sujet_list_project_wise[project]:
                    for chan_i, chan in enumerate(_xr_data_allsujet['chan']): 
                        plt.plot(zscore(_xr_data_allsujet.loc[sujet,chan,:]) + chan_i)
                    plt.title(sujet)
                    plt.show()

                    hzPxx, Pxx = scipy.signal.welch(_xr_data_allsujet.loc[sujet,chan,:], fs=srate, window='hann', nperseg=srate*20, noverlap=srate*10, nfft=None)
                    plt.semilogy(hzPxx, Pxx)
                    plt.show()

            os.chdir(os.path.join(path_prep, 'data_aggregates'))
            _xr_data_allsujet.to_netcdf(f"{project}_raw.nc")
                    
        ######## NORMATIVE ########

        if project == 'NORMATIVE':

            #cond = condition_list_project_wise[project][-3]
            for cond in condition_list_project_wise[project]:

                #_sujet_i, _sujet = 0, sujet_list_project_wise[project][0]
                for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                    os.chdir(os.path.join(path_data, project, 'first', _sujet))

                    print(f"OPEN {project} : {_sujet}")

                    if os.path.exists(os.path.join(path_data, project, 'first', _sujet, f"{_sujet}_{cond}_ValidICM.vhdr")) == False:

                        continue

                    else:

                        _data = mne.io.read_raw_brainvision(f"{_sujet}_{cond}_ValidICM.vhdr")
                        _data_extract = _data.get_data()
                        _srate = _data.info['sfreq']
                        _chan_list = _data.info['ch_names']
                        _lowpass = _data.info['lowpass']

                        if debug:
                            mne.viz.plot_raw(_data, n_channels=1)

        ######## PHYSIOLOGY ########        

        if project == 'PHYSIOLOGY':

            #_sujet_i, _sujet = 1, sujet_list_project_wise[project][6]
            for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                if _sujet in ['MC05', 'OL04']:
                    continue

                os.chdir(os.path.join(path_data, project, _sujet))

                print(f"OPEN {project} : {_sujet}")
                
                _data = mne.io.read_raw_brainvision(f"{_sujet}_CONTINU_64Ch_A2Ref.vhdr")
                _data_extract = _data.get_data()
                _srate = _data.info['sfreq']
                _chan_list = _data.info['ch_names']
                _lowpass = _data.info['lowpass']
                trig_cond = {_trig : _time for _trig, _time in zip(_data.annotations.description, _data.annotations.onset) if _trig.find('Comment') != -1}
                _respi = _data_extract[chan_list_project_wise[project].index('Pression'),:]

                if debug:
                    mne.viz.plot_raw(_data, n_channels=1, event_id=trig_cond)

                    plt.plot(_respi)
                    plt.vlines(np.array(list(trig_cond.values()))*_srate, ymin=_respi.min(), ymax=_respi.max(), color='r')
                    plt.title(_sujet)
                    plt.show()


        ######## SLP ########        

        if project == 'SLP':

            #_sujet_i, _sujet = 0, sujet_list_project_wise[project][0]
            for _sujet_i, _sujet in enumerate(sujet_list_project_wise[project]):

                os.chdir(os.path.join(path_data, project, _sujet))

                print(f"OPEN {project} : {_sujet}")
                
                _data = mne.io.read_raw_brainvision(f"64Ch_SLP_{_sujet}_A2Ref.vhdr")
                _data_extract = _data.get_data()
                _srate = _data.info['sfreq']
                _chan_list = _data.info['ch_names']
                _lowpass = _data.info['lowpass']

                if debug:
                    mne.viz.plot_raw(_data, n_channels=1)

                    plt.plot(_respi)
                    plt.vlines(np.array(list(trig_cond.values()))*_srate, ymin=_respi.min(), ymax=_respi.max(), color='r')
                    plt.title(_sujet)
                    plt.show()


        ######## ITL LEO ########        

        if project == 'ITL_LEO':

            os.chdir(os.path.join(path_data, 'ITL_LEO'))

            #_sujet = sujet_list_project_wise[project][0]
            for _sujet in sujet_list_project_wise[project]:

                #cond = condition_list_project_wise[project][0]
                for cond in condition_list_project_wise[project]:

                    file_name = [file for file in os.listdir() if file.find(_sujet) != -1 and file.find(f'{cond}.edf') != -1][0]
                    file_name_marker = [file for file in os.listdir() if file.find(_sujet) != -1 and file.find(f'{cond}.Markers') != -1][0]
                    
                    _data = mne.io.read_raw_edf(file_name)
                    _data_extract = _data.get_data()
                    _srate = _data.info['sfreq']
                    _chan_list = _data.info['ch_names']
                    _lowpass = _data.info['lowpass']

                    if debug:
                        mne.viz.plot_raw(_data, n_channels=1)

                        plt.plot(_respi)
                        plt.vlines(np.array(list(trig_cond.values()))*_srate, ymin=_respi.min(), ymax=_respi.max(), color='r')
                        plt.title(_sujet)
                        plt.show()


def compare_chan_list():

    chan_list_allsujet = {}

    cond = 'VS'

    for sujet in sujet_list:

        ######## identify project and sujet ########        
        sujet_project = sujet_project_nomenclature[sujet[2:4]]
        sujet_init_name = list(sujet_list_correspondance.keys())[list(sujet_list_correspondance.values()).index(sujet)]

        ######## OPEN DATA ########
        if sujet_project == 'NORMATIVE':

            os.chdir(os.path.join(path_data, sujet_project, 'first', sujet_init_name))
            _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_{cond}_ValidICM.vhdr")
            _chan_list = _data.info['ch_names']

        elif sujet_project == 'PHYSIOLOGY':

            os.chdir(os.path.join(path_data, sujet_project, sujet_init_name))
            _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_CONTINU_64Ch_A2Ref.vhdr")
            _chan_list_eeg = _data.info['ch_names']

        elif sujet_project == 'ITL_LEO':

            os.chdir(os.path.join(path_data, 'ITL_LEO'))
            file_name = [file for file in os.listdir() if file.find(sujet_init_name) != -1 and file.find(f'{cond}.edf') != -1][0]
            _data = mne.io.read_raw_edf(file_name)
            _chan_list_eeg = _data.info['ch_names']

        chan_list_allsujet[sujet] = _chan_list_eeg
            




################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    export_all_df_alldata()

    explore_all_data()

    compare_chan_list()


