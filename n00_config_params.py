
import numpy as np
import scipy.signal



################################
######## GENERAL PARAMS ######## 
################################

teleworking = False

enable_big_execute = False
perso_repo_computation = False

srate_g = 500

project_name_list_raw = ['COVEM_ITL', 'NORMATIVE', 'PHYSIOLOGY', 'SLP', 'ITL_LEO']
project_name_list = ['NORMATIVE', 'PHYSIOLOGY', 'ITL_LEO']

sujet_list_project_wise = {'COVEM_ITL': ['01NM', '02HM', '03DG', '04DM', '05DR', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '13NN', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT'],
                      'NORMATIVE' : ['MW02', 'OL04', 'MC05', 'VS06', 'LS07', 'JS08', 'HC09', 'YB10','ML11', 'CM12', 'CV13', 'VA14', 'LC15', 'PS16', 'SL18', 'JP19', 'LD20'], 
                      'PHYSIOLOGY' : ['JS08',  'LP26',  'MN23',  'SB27',  'TH24',  'VA14',  'VS06'], 
                      'SLP' : ['AB33', 'BK35', 'CD28', 'ES32', 'JC30', 'MM34', 'SG29', 'ZM31'],
                      'ITL_LEO' : ['01NM', '02HM', '03DG', '04DM', '05DR', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '13NN', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT']}

sujet_list = ['01NM_MW', '02NM_OL', '03NM_MC', '04NM_VS', '05NM_LS', '06NM_JS', '07NM_HC', '08NM_YB','09NM_ML', '10NM_CM', '11NM_CV', '12NM_VA', '13NM_LC', '14NM_PS', '15NM_SL', '16NM_JP', '17NM_LD',
              '18PH_JS',  '19PH_LP',  '20PH_MN',  '21PH_SB',  '22PH_TH',  '23PH_VA',  '24PH_VS',
              '25IL_NM', '26IL_HM', '27IL_DG', '28IL_DM', '29IL_DR', '30IL_DJ', '31IL_DC', '32IL_AP', '33IL_SL', '34IL_LL', '35IL_VR', '36IL_LC', '37IL_NN', '38IL_MA', '39IL_LY', '40IL_BA', '41IL_CM', '42IL_EA', '43IL_LT']

cond_list = ['VS', 'CHARGE']

sujet_project_nomenclature = {'NM' : 'NORMATIVE', 'PH' : 'PHYSIOLOGY', 'IL' : 'ITL_LEO'}

sujet_list_correspondance = {'MW02' : '01NM_MW', 'OL04' : '02NM_OL', 'MC05' : '03NM_MC', 'VS06' : '04NM_VS', 'LS07' : '05NM_LS', 'JS08' : '06NM_JS', 'HC09' : '07NM_HC', 
                             'YB10' : '08NM_YB', 'ML11' : '09NM_ML', 'CM12' : '10NM_CM', 'CV13' : '11NM_CV', 'VA14' : '12NM_VA', 'LC15' : '13NM_LC', 'PS16' : '14NM_PS', 
                             'SL18' : '15NM_SL', 'JP19' : '16NM_JP', 'LD20' : '17NM_LD', 'JS08' : '18PH_JS', 'LP26' : '19PH_LP', 'MN23' : '20PH_MN', 'SB27' : '21PH_SB',
                             'TH24' : '22PH_TH', 'VA14' : '23PH_VA', 'VS06' : '24PH_VS', '01NM' : '25IL_NM', '02HM' : '26IL_HM', '03DG' : '27IL_DG', '04DM' : '28IL_DM', 
                             '05DR' : '29IL_DR', '06DJ' : '30IL_DJ', '07DC' : '31IL_DC', '08AP' : '32IL_AP', '09SL' : '33IL_SL', '10LL' : '34IL_LL', '11VR' : '35IL_VR', 
                             '12LC' : '36IL_LC', '13NN' : '37IL_NN', '14MA' : '38IL_MA', '15LY' : '39IL_LY', '16BA' : '40IL_BA', '17CM' : '41IL_CM', '18EA' : '42IL_EA', 
                             '19LT' : '43IL_LT'}

chan_list_project_wise = {'COVEM_ITL': ['FC1', 'FC2', 'Cz', 'C2', 'CP1', 'CP2', 'EMG'],
                      'NORMATIVE' : ['FP1', 'F7', 'F3', 'Fz', 'FC5', 'FC1', 'A1', 'T7', 'C3', 'Cz', 'TP9', 'CP5', 'CP1', 'P7', 'P3', 'Pz', 'FP2', 'F4', 'F8', 'FC2', 'FC6', 'C4', 'T8', 'A2', 'CP2', 'CP6', 'TP10', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Debit', 'Pression', 'EMG PS', 'ECG', 'FCz'], 
                      'PHYSIOLOGY' : ['EOG', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Debit', 'Pression', 'PS', 'ECG'], 
                      'SLP' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'ECG', 'ScalEMG'],
                      'ITL_LEO' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'A1', 'CP5', 'CP1', 'CP2', 'CP6', 'A2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'EOG', 'EMG', 'PRESSION']}



#### NOTES ####
# In PHYSIOLOGY sujet ['MC05', 'OL04'] have been excluded because they cant load

condition_list_project_wise = {'COVEM_ITL': ['CHARGE'],
                      'NORMATIVE' : ['CHARGE', 'PETITE CHARGE', 'ARTIFACT', 'SNIFS', 'VS', 'VS2'], 
                      'PHYSIOLOGY' : ['CHARGE'], 
                      'SLP' : ['CHARGE'],
                       'ITL_LEO' : ['VS', 'CO2', 'ITL']}

params_extraction_data = {'COVEM_ITL' : {'time_cutoff' : 13},
                      'NORMATIVE' : { 'time_cutoff' : {'CHARGE' : 0, 'SNIFS' : 0, 'VS' : 0}},
                      'PHYSIOLOGY' : {'time_cutoff' : 0},
                      'SLP' : {'time_cutoff' : 0},
                       'ITL_LEO' : {'time_cutoff' : 0}}

physiology_trig = {'JS08' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 323500], 'stop' : [300000, 642000]},
                   'LP26' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 350000], 'stop' : [300000, 668000]},
                   'MN23' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 330000], 'stop' : [300000, 663000]},
                   'SB27' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 342000], 'stop' : [300000, 650000]},
                   'TH24' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 330000], 'stop' : [300000, 633000]},
                   'VA14' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 316000], 'stop' : [310000, 627000]},
                   'VS06' : {'trig' : ['VS', 'CHARGE'], 'start' : [0, 362000], 'stop' : [310000, 670000]}}


########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()

if PC_ID == 'LAPTOP-EI7OSP7K':

    if teleworking:

        PC_working = 'Jules_VPN'
        if perso_repo_computation:
            path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
        else:    
            path_main_workdir = 'Z:\\Projets\\PPI_Jules\\Scripts'
        path_general = 'Z:\\Projets\\PPI_Jules'
        path_memmap = 'Z:\\Projets\\PPI_Jules\\Mmap'
        n_core = 4

    else:

        PC_working = 'Jules_VPN'
        if perso_repo_computation:
            path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
        else:    
            path_main_workdir = 'N:\\cmo\Projets\\PPI_Jules\\Scripts'
        path_general = 'N:\\cmo\\Projets\\PPI_Jules'
        path_memmap = 'N:\\cmo\\Projets\\PPI_Jules\\Mmap'
        n_core = 4

    

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    if perso_repo_computation:
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    else:    
        path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\Mmap'
    n_core = 2

elif PC_ID == 'pc-jules' or PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/jules/Bureau/perso_repo_computation/Scripts'
    else:    
        path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/Mmap'
    n_core = 4

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    if perso_repo_computation:
        path_main_workdir = '/home/valentin/Bureau/perso_repo_computation/Script_Python_EEG_Paris_git'
    else:    
        path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Mmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 15

else:

    PC_working = 'crnl_cluster'
    path_main_workdir = '/crnldata/cmo/Projets/PPI_Jules/Scripts'
    path_general = '/crnldata/cmo/Projets/PPI_Jules'
    path_memmap = '/mnt/data/julesgranget'
    n_core = 10
    

path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_slurm = os.path.join(path_general, 'Script_slurm')

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10







################################
######## RESPI PARAMS ########
################################ 

#### INSPI DOWN
sujet_respi_adjust = {
'01PD':'inverse',   '02MJ':'inverse',   '03VN':'inverse',   '04GB':'inverse',   '05LV':'inverse',
'06EF':'inverse',   '07PB':'inverse',   '08DM':'inverse',   '09TA':'inverse',   '10BH':'inverse',
'11FA':'inverse',   '12BD':'inverse',   '13FP':'inverse',   '14MD':'inverse',   '15LG':'inverse',
'16GM':'inverse',   '17JR':'inverse',   '18SE':'inverse',   '19TM':'inverse',   '20TY':'inverse',
'21ZV':'inverse',   '22DI':'inverse',   '23LF':'inverse',   '24TJ':'inverse',   '25DF':'inverse',
'26MN':'inverse',   '27BD':'inverse',   '28NT':'inverse',   '29SC':'inverse',   '30AR':'inverse',
'31HJ':'inverse',   '32CM':'inverse',   '33MA':'inverse'
}


cycle_detection_params = {
'exclusion_metrics' : 'med',
'metric_coeff_exclusion' : 3,
'inspi_coeff_exclusion' : 2,
'respi_scale' : [0.1, 0.35], #Hz
}


scale_for_respi_abnormalities = {'04GB' : {'session' : 'o', 'coeff' : 6840}, '07PB' : {'session' : 'o', 'coeff' : 597}}


################################
######## ECG PARAMS ########
################################ 

sujet_ecg_adjust = {
'01PD':'inverse',   '02MJ':'inverse',   '03VN':'inverse',   '04GB':'inverse',   '05LV':'inverse',
'06EF':'inverse',   '07PB':'inverse',   '08DM':'inverse',   '09TA':'inverse',   '10BH':'inverse',
'11FA':'inverse',   '12BD':'inverse',   '13FP':'inverse',   '14MD':'inverse',   '15LG':'inverse',
'16GM':'inverse',   '17JR':'inverse',   '18SE':'inverse',   '19TM':'inverse',   '20TY':'inverse',
'21ZV':'inverse',   '22DI':'inverse',   '23LF':'inverse',   '24TJ':'inverse',   '25DF':'inverse',
'26MN':'inverse',   '27BD':'inverse',   '28NT':'inverse',   '29SC':'inverse',   '30AR':'inverse',
'31HJ':'inverse',   '32CM':'inverse',   '33MA':'inverse'
}


hrv_metrics_short_name = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_SD1', 'HRV_SD2']




################################
######## PREP PARAMS ########
################################ 


prep_step_debug = {
'reref' : {'execute': True, 'params' : ['TP9']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': True},
'csd_computation' : {'execute': True},
}

prep_step_wb = {
'reref' : {'execute': False, 'params' : ['TP9', 'TP10']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'csd_computation' : {'execute': False},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
}

prep_step_lf = {
'reref' : {'execute': False, 'params' : ['chan']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}

prep_step_hf = {
'reref_mastoide' : {'execute': False},
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : 55, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : None, 'h_freq': None}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': True},
}





################################
######## ERP PARAMS ########
################################



PPI_time_vec = [-2.5, 1] #seconds
ERP_time_vec = [-2.5, 2.5]
mean_respi_ERP_time_vec = [-3,3]
PPI_lm_time = [-2.5, 0]

allplot_erp_ylim = (-0.3, 0.3)

ERP_n_surrogate = 1000




########################################
######## PARAMS SURROGATES ########
########################################

#### Pxx Cxy

zero_pad_coeff = 15

def get_params_spectral_analysis(srate):
    nwind = int( 20*srate ) # window length in seconds*srate
    nfft = nwind*zero_pad_coeff # if no zero padding nfft = nwind
    noverlap = np.round(nwind/2) # number of points of overlap here 50%
    hannw = scipy.signal.windows.hann(nwind) # hann window

    return nwind, nfft, noverlap, hannw

#### plot Pxx Cxy  
if zero_pad_coeff - 5 <= 0:
    remove_zero_pad = 0
remove_zero_pad = zero_pad_coeff - 5

#### stretch
stretch_point_surrogates = 500

#### coh
n_surrogates_coh = 500
freq_surrogates = [0, 2]
percentile_coh = .95

#### cycle freq
n_surrogates_cyclefreq = 500
percentile_cyclefreq_up = .99
percentile_cyclefreq_dw = .01






################################
######## PRECOMPUTE TF ########
################################


#### stretch
stretch_point_TF = 500
stretch_TF_auto = False
ratio_stretch_TF = 0.5

#### TF & ITPC
nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
srate_dw = 10
wavetime = np.arange(-3,3,1/srate_g)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')
Pxx_wavelet_norm = 1000


#### STATS
n_surrogates_tf = 500
tf_percentile_sel_stats_dw = 5 
tf_percentile_sel_stats_up = 95 
tf_stats_percentile_cluster = 95
tf_stats_percentile_cluster_manual_perm = 80
erp_time_cluster_thresh = 50 #ms
norm_method = 'rscore'# 'zscore', 'dB'
exclude_frex_range = [48, 52]

#### plot
tf_plot_percentile_scale = 99 #for one side




################################
######## POWER ANALYSIS ########
################################

#### analysis
coh_computation_interval = .02 #Hz around respi


################################
######## FC ANALYSIS ########
################################

nfrex_fc = 50

#### band to remove
freq_band_fc_analysis = {'theta' : [4, 8], 'alpha' : [9,12], 'beta' : [15,40], 'l_gamma' : [50, 80], 'h_gamma' : [80, 120]}

percentile_thresh = 90

#### for DFC
slwin_dict = {'theta' : 5, 'alpha' : 3, 'beta' : 1, 'l_gamma' : .3, 'h_gamma' : .3} # seconds
slwin_step_coeff = .1  # in %, 10% move

band_name_fc_dfc = ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma']

#### cond definition
cond_FC_DFC = ['FR_CV', 'AL', 'SNIFF', 'AC']

#### down sample for AL
dw_srate_fc_AL = 10

#### down sample for AC
dw_srate_fc_AC = 50

#### n points for AL interpolation
n_points_AL_interpolation = 10000
n_points_AL_chunk = 1000

#### for df computation
percentile_graph_metric = 25



################################
######## TOPOPLOT ########
################################

around_respi_Cxy = 0.025


################################
######## HRV ANALYSIS ########
################################



srate_resample_hrv = 10
nwind_hrv = int( 128*srate_resample_hrv )
nfft_hrv = nwind_hrv
noverlap_hrv = np.round(nwind_hrv/10)
win_hrv = scipy.signal.windows.hann(nwind_hrv)
f_RRI = (.1, .5)




