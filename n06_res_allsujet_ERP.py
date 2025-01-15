

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n04_precompute_ERP import *

debug = False





################################
######## ERP PLOT ########
################################



def plot_ERP_mean_subject_wise(stretch=False):

    print('ERP PLOT', flush=True)

    t_start_ERP = ERP_time_vec[0]
    t_stop_ERP = ERP_time_vec[1]

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats = get_cluster_stats_manual_prem_subject_wise(stretch=True)
    
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats = get_cluster_stats_manual_prem_subject_wise(stretch=False)

    if stretch:
        time_vec = np.arange(stretch_point_ERP)
    else:
        time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)

    ######## IDENTIFY MIN MAX ########

    print('PLOT SUMMARY ERP')

    #### identify min max for allsujet

    _absmax = {}

    #nchan_i, nchan = 1, chan_list_eeg[1]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        _absmax_chan = np.array([])

        #sujet = sujet_list[16]
        for sujet in sujet_list:

            #cond_i, cond = 2, conditions_diff[2]
            for cond_i, cond in enumerate(cond_list):

                data_stretch = xr_data.loc[sujet, cond, nchan, :]
                sem = xr_data_sem.loc[sujet, cond, nchan, :]
                data_stretch_up, data_stretch_down = data_stretch + sem, data_stretch - sem
                _absmax_chan = np.append(_absmax_chan, np.array([data_stretch_up.max().values, data_stretch_down.min().values]))

        _absmax[nchan] = np.abs(_absmax_chan).max()

    if debug:

        for sujet in sujet_list:

            plt.plot(sem.loc[sujet, 'VS', nchan, :])
        
        plt.show()

    #sujet_i, sujet = 0, sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=2)

            fig.set_figheight(8)
            fig.set_figwidth(8)

            if stretch:
                plt.suptitle(f'stretch {nchan} {sujet}')
            else:
                plt.suptitle(f'{nchan} {sujet}')

            for cond_i, cond in enumerate(cond_list):

                ax = axs[cond_i]
                data_stretch = xr_data.loc[sujet, cond, nchan, :]
                sem = xr_data.loc[sujet, cond, nchan, :].std()

                ax.set_ylim(-_absmax[nchan], _absmax[nchan])
                ax.set_title(cond)

                ax.plot(time_vec, data_stretch,color='r')
                ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                clusters = cluster_stats.loc[sujet, cond, nchan, :].values
                ax.fill_between(time_vec, -_absmax[nchan], _absmax[nchan], where=clusters.astype('int'), alpha=0.3, color='r')

                ax.invert_yaxis()

                if stretch:
                    ax.vlines(stretch_point_ERP/2, ymin=-_absmax[nchan], ymax=_absmax[nchan], colors='g')  
                else:
                    ax.vlines(0, ymin=-_absmax[nchan], ymax=_absmax[nchan], colors='g')  

            fig.tight_layout()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'ERP', 'summary_diff_subject_wise', 'allchan'))

            if stretch:

                fig.savefig(f'stretch_{nchan}_{sujet}.jpeg', dpi=150)

            else:

                fig.savefig(f'nostretch_{nchan}_{sujet}.jpeg', dpi=150)

            if nchan in chan_list_eeg_short:

                os.chdir(os.path.join(path_results, 'ERP', 'summary_diff_subject_wise'))

                if stretch:

                    fig.savefig(f'stretch_{nchan}_{sujet}.jpeg', dpi=150)

                else:

                    fig.savefig(f'nostretch_{nchan}_{sujet}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()


    


def plot_ERP_mean_allsujet(stretch=False):

    print('ERP PLOT', flush=True)

    t_start_ERP = ERP_time_vec[0]
    t_stop_ERP = ERP_time_vec[1]

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=True)
    
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats = get_cluster_stats_manual_prem_allsujet(stretch=False)

    if stretch:
        time_vec = np.arange(stretch_point_ERP)
    else:
        time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)
        mask_time_vec_zoomin = (time_vec >= -0.5) & (time_vec < 0.5) 

    ######## IDENTIFY MIN MAX ########

    print('PLOT SUMMARY ERP')

    #### identify min max for allsujet

    _absmax = np.array([])

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg_short):

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(cond_list):

            data_stretch = xr_data.loc[:, cond, nchan, :].mean('sujet').values
            sem = xr_data.loc[:, cond, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, nchan, :].shape[0])
            data_stretch_up, data_stretch_down = data_stretch + sem, data_stretch - sem
            _absmax = np.concatenate([_absmax, np.abs(data_stretch_up), np.abs(data_stretch_down)])

    absmax_group = _absmax.max()

    if debug:

        for sujet in sujet_list:

            plt.plot(xr_data.loc[sujet, 'CHARGE', nchan, :])
        
        plt.show()

    n_sujet = xr_data.loc[:, cond, nchan, :].shape[0]

    #nchan_i, nchan = 0, chan_list_eeg_short[0]
    for nchan_i, nchan in enumerate(chan_list_eeg_short):

        fig, ax = plt.subplots()

        fig.set_figheight(5)
        fig.set_figwidth(8)

        if stretch:
            plt.suptitle(f'stretch {nchan} nsujet:{n_sujet}')
        else:
            plt.suptitle(f'{nchan} nsujet:{n_sujet}')

        data_stretch = xr_data.loc[:, 'CHARGE', nchan, :].mean('sujet').values
        sem = xr_data.loc[:, 'CHARGE', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'CHARGE', nchan, :].shape[0])
        baseline = xr_data.loc[:, 'VS', nchan, :].mean('sujet').values
        sem_baseline = xr_data.loc[:, 'VS', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'VS', nchan, :].shape[0])

        ax.set_ylim(-absmax_group, absmax_group)

        ax.plot(time_vec, data_stretch, label='CHARGE',color='r')
        ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

        ax.plot(time_vec, baseline, label='VS', color='b')
        ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

        clusters = cluster_stats.loc[nchan, :].values
        ax.fill_between(time_vec, -absmax_group, absmax_group, where=clusters.astype('int'), alpha=0.3, color='r')

        ax.invert_yaxis()

        if stretch:
            ax.vlines(stretch_point_ERP/2, ymin=-absmax_group, ymax=absmax_group, colors='g')  
        else:
            ax.vlines(0, ymin=-absmax_group, ymax=absmax_group, colors='g')  

        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'ERP', 'summary_diff_allsujet', 'allchan'))

        if stretch:

            fig.savefig(f'stretch_{nchan}.jpeg', dpi=150)

        else:

            fig.savefig(f'nostretch_{nchan}.jpeg', dpi=150)

        if nchan in chan_list_eeg_short:

            os.chdir(os.path.join(path_results, 'ERP', 'summary_diff_allsujet'))

            if stretch:

                fig.savefig(f'stretch_{nchan}.jpeg', dpi=150)

            else:

                fig.savefig(f'nostretch_{nchan}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()









################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### diff for allsujet
    plot_ERP_mean_allsujet(stretch=False)
    plot_ERP_mean_allsujet(stretch=True)

    plot_ERP_mean_subject_wise(stretch=False)
    plot_ERP_mean_subject_wise(stretch=True)









