
import numpy as np
import scipy.signal



################################
######## GENERAL PARAMS ######## 
################################

teleworking = False

enable_big_execute = False
perso_repo_computation = False

srate = 500

project_name_list_raw = ['COVEM_ITL', 'NORMATIVE', 'PHYSIOLOGY', 'SLP', 'ITL_LEO']
project_name_list = ['NORMATIVE', 'PHYSIOLOGY', 'ITL_LEO']

sujet_list_project_wise = {'COVEM_ITL': ['01NM', '02HM', '03DG', '04DM', '05DR', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '13NN', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT'],
                      'NORMATIVE' : ['MW02', 'OL04', 'MC05', 'LS07', 'JS08', 'HC09', 'YB10', 'CM12', 'CV13', 'VA14', 'LC15', 'PS16', 'JP19', 'LD20'], 
                      'PHYSIOLOGY' : ['JS08',  'LP26',  'MN23',  'SB27',  'TH24',  'VA14',  'VS06'], 
                      'SLP' : ['AB33', 'BK35', 'CD28', 'ES32', 'JC30', 'MM34', 'SG29', 'ZM31'],
                      'ITL_LEO' : ['01NM', '03DG', '04DM', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT']}

sujet_list = ['01NM_MW', '02NM_OL', '03NM_MC', '04NM_LS', '05NM_JS', '06NM_HC', '07NM_YB', '08NM_CM', '09NM_CV', '10NM_VA', '11NM_LC', '12NM_PS', '13NM_JP', '14NM_LD',
              '15PH_JS',  '16PH_LP',  '17PH_MN',  '18PH_SB',  '19PH_TH',  '20PH_VA',  '21PH_VS',
              '22IL_NM', '23IL_DG', '24IL_DM', '25IL_DJ', '26IL_DC', '27IL_AP', '28IL_SL', '29IL_LL', '30IL_VR', '31IL_LC', '32IL_MA', '33IL_LY', '34IL_BA', '35IL_CM', '36IL_EA', '37IL_LT']

cond_list = ['VS', 'CHARGE']

sujet_project_nomenclature = {'NM' : 'NORMATIVE', 'PH' : 'PHYSIOLOGY', 'IL' : 'ITL_LEO'}

sujet_list_correspondance = {'NM_MW02' : '01NM_MW', 'NM_OL04' : '02NM_OL', 'NM_MC05' : '03NM_MC', 'NM_LS07' : '04NM_LS', 'NM_JS08' : '05NM_JS', 'NM_HC09' : '06NM_HC', 
                             'NM_YB10' : '07NM_YB', 'NM_CM12' : '08NM_CM', 'NM_CV13' : '09NM_CV', 'NM_VA14' : '10NM_VA', 'NM_LC15' : '11NM_LC', 'NM_PS16' : '12NM_PS', 
                             'NM_JP19' : '13NM_JP', 'NM_LD20' : '14NM_LD', 'PH_JS08' : '15PH_JS', 'PH_LP26' : '16PH_LP', 'PH_MN23' : '17PH_MN', 'PH_SB27' : '18PH_SB',
                             'PH_TH24' : '19PH_TH', 'PH_VA14' : '20PH_VA', 'PH_VS06' : '21PH_VS', 'IL_01NM' : '22IL_NM', 'IL_03DG' : '23IL_DG', 'IL_04DM' : '24IL_DM', 
                             'IL_06DJ' : '25IL_DJ', 'IL_07DC' : '26IL_DC', 'IL_08AP' : '27IL_AP', 'IL_09SL' : '28IL_SL', 'IL_10LL' : '29IL_LL', 'IL_11VR' : '30IL_VR', 
                             'IL_12LC' : '31IL_LC', 'IL_14MA' : '32IL_MA', 'IL_15LY' : '33IL_LY', 'IL_16BA' : '34IL_BA', 'IL_17CM' : '35IL_CM', 'IL_18EA' : '36IL_EA', 'IL_19LT' : '37IL_LT'}

chan_list_project_wise = {'COVEM_ITL': ['FC1', 'FC2', 'Cz', 'C2', 'CP1', 'CP2', 'EMG'],
                      'NORMATIVE' : ['Fp1', 'F7', 'F3', 'Fz', 'FC5', 'FC1', 'A1', 'T7', 'C3', 'Cz', 'TP9', 'CP5', 'CP1', 'P7', 'P3', 'Pz', 'Fp2', 'F4', 'F8', 'FC2', 'FC6', 'C4', 'T8', 'A2', 'CP2', 'CP6', 'TP10', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Debit', 'Pression', 'EMG PS', 'ECG', 'FCz'], 
                      'PHYSIOLOGY' : ['EOG', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Debit', 'Pression', 'PS', 'ECG'], 
                      'SLP' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'ECG', 'ScalEMG'],
                      'ITL_LEO' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'A1', 'CP5', 'CP1', 'CP2', 'CP6', 'A2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'EOG', 'EMG', 'PRESSION']}

chan_list = np.array(['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8', 'pression'])

chan_list_eeg = np.array(['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8'])

#### NOTES ####
# In PHYSIOLOGY sujet ['MC05', 'OL04'] have been excluded because they cant load
# In ITL sujet ['NN', ] have been excluded due to bad signals in ITL
# In NORMATIVE ['VS06', 'ML11', 'SL18', 'DR05'] removed no signal in CHARGE

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



########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()
init_workdir = os.getcwd()

if PC_ID == 'LAPTOP-EI7OSP7K':

    try: 
        os.chdir('N:\\')
        teleworking = False
    except:
        teleworking = True 

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

os.chdir(init_workdir)

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10







################################
######## RESPI PARAMS ########
################################ 


sujet_respi_adjust = {
'01NM_MW':'normal',   '02NM_OL':'normal',   '03NM_MC':'normal',   '04NM_LS':'normal',
'05NM_JS':'normal',   '06NM_HC':'normal',   '07NM_YB':'normal',   '08NM_CM':'normal',
'09NM_CV':'normal',   '10NM_VA':'normal',   '11NM_LC':'normal',   '12NM_PS':'normal',  
'13NM_JP':'normal',   '14NM_LD':'normal',   '15PH_JS':'inverse',   '16PH_LP':'normal',   '17PH_MN':'inverse',
'18PH_SB':'normal',   '19PH_TH':'inverse',   '20PH_VA':'inverse',   '21PH_VS':'inverse',   '22IL_NM':'inverse',
'23IL_DG':'inverse',   '24IL_DM':'inverse',   '25IL_DJ':'inverse',
'26IL_DC':'inverse',   '27IL_AP':'inverse',   '28IL_SL':'inverse',   '29IL_LL':'inverse',   '30IL_VR':'inverse',
'31IL_LC':'inverse',   '32IL_MA':'inverse',   '33IL_LY':'inverse',   '34IL_BA':'inverse',   '35IL_CM':'inverse',   
'36IL_EA':'inverse',   '37IL_LT':'inverse'
}


cycle_detection_params = {
'exclusion_metrics' : 'med',
'sum_coeff_exclusion' : 3,
'time_coeff_exclusion' : 2,
'respi_scale' : [0.1, 0.35], #Hz
}





################################
######## PREP PARAMS ########
################################ 

section_time_general = 300 #sec

section_timming_PHYSIOLOGY = {
'15PH_JS': {'VS' : [0, 613], 'CHARGE' : [644, 1280]},   '16PH_LP': {'VS' : [0, 610], 'CHARGE' : [700, 1340]},   '17PH_MN': {'VS' : [0, 610], 'CHARGE' : [655, 1340]}, 
'18PH_SB': {'VS' : [0, 610], 'CHARGE' : [680, 1311]},   '19PH_TH': {'VS' : [0, 620], 'CHARGE' : [650, 1280]},   '20PH_VA': {'VS' : [0, 610], 'CHARGE' : [630, 1250]},   
'21PH_VS': {'VS' : [0, 635], 'CHARGE' : [720, 1340]}
}


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

prep_step = {
'reref' : {'execute': False, 'params' : ['TP9', 'TP10']}, #chan = chan to reref
'detrend_mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'csd_computation' : {'execute': False},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
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



