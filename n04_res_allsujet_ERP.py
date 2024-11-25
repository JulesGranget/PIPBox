

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n04bis_res_allsujet_ERP import *

debug = False







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    ######## LOAD DATA ########

    xr_data, xr_data_sem = compute_ERP()
    xr_data_stretch, xr_data_sem_stretch = compute_ERP_stretch()
    
    ######## STATS ########
    
    cluster_stats = get_cluster_stats_manual_prem(stretch=False)
    cluster_stats_stretch = get_cluster_stats_manual_prem(stretch=True)

    get_cluster_stats_manual_prem_one_cond()
    get_cluster_stats_manual_prem_one_cond_stretch()

    plot_ERP_diff(stretch=False)
    plot_ERP_diff(stretch=True)

    plot_ERP_mean_subject_wise(stretch=False)
    plot_ERP_mean_subject_wise(stretch=True)






