
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc
from matplotlib.animation import FuncAnimation
import networkx as nx

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False






################################
######## PLOT FC ########
################################


def plot_allsujet_FC_chunk():

    for stretch in [True, False]:

        #fc_metric = 'MI'
        for fc_metric in ['MI', 'ISPC', 'WPLI']:

            print(f'{fc_metric} PLOT stretch:{stretch}', flush=True)

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
            if stretch:
                fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
                clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_stretch.nc')
            else:
                fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
                clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS.nc')

            pairs_to_compute = fc_allsujet['pair'].values
            time_vec = fc_allsujet['time'].values

            ######## IDENTIFY MIN MAX ########

            #### identify min max for allsujet

            vlim_band = {}

            for band in freq_band_fc_list:
            
                vlim = np.array([])

                #pair_i, pair = 0, pairs_to_compute[0]
                for pair_i, pair in enumerate(pairs_to_compute):

                    #cond_i, cond = 0, cond_list[0]
                    for cond_i, cond in enumerate(cond_list):

                        if fc_metric == 'MI':
                            data_chunk = fc_allsujet.loc[:, pair, cond, :].mean('sujet').values
                            sem = fc_allsujet.loc[:, pair, cond, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, cond, :].shape[0])
                        else:
                            data_chunk = fc_allsujet.loc[:, band, cond, pair, :].mean('sujet').values
                            sem = fc_allsujet.loc[:, band, cond, pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, cond, pair, :].shape[0])
                                
                        data_chunk_up, data_chunk_down = data_chunk + sem, data_chunk - sem
                        vlim = np.concatenate([vlim, data_chunk_up, data_chunk_down])

                vlim = {'min' : vlim.min(), 'max' : vlim.max()}
                vlim_band[band] = vlim

            if debug:

                for pair in pairs_to_compute:

                    plt.plot(fc_allsujet.loc[:, 'CHARGE', pair, :].mean('sujet'))
                
                plt.show()

            n_sujet = fc_allsujet['sujet'].shape[0]

            for band in freq_band_fc_list:

                #pair_i, pair = 0, pairs_to_compute[0]
                for pair_i, pair in enumerate(pairs_to_compute):

                    fig, ax = plt.subplots()

                    fig.set_figheight(5)
                    fig.set_figwidth(8)

                    if stretch:
                        plt.suptitle(f'stretch {pair} nsujet:{n_sujet}')
                    else:
                        plt.suptitle(f'{pair} nsujet:{n_sujet}')

                    if fc_metric == 'MI':
                        data_chunk = fc_allsujet.loc[:, pair, 'CHARGE', :].mean('sujet').values
                        sem = fc_allsujet.loc[:, pair, 'CHARGE', :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, 'CHARGE', :].shape[0])
                        baseline = fc_allsujet.loc[:, pair, 'VS', :].mean('sujet').values
                        sem_baseline = fc_allsujet.loc[:, pair, 'VS', :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, pair, 'VS', :].shape[0])
                    else:
                        data_chunk = fc_allsujet.loc[:, band, 'CHARGE', pair, :].mean('sujet').values
                        sem = fc_allsujet.loc[:, band, 'CHARGE', pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, 'CHARGE', pair, :].shape[0])
                        baseline = fc_allsujet.loc[:, band, 'VS', pair, :].mean('sujet').values
                        sem_baseline = fc_allsujet.loc[:, band, 'VS', pair, :].std('sujet').values / np.sqrt(fc_allsujet.loc[:, band, 'VS', pair, :].shape[0])

                    ax.set_ylim(vlim_band[band]['min'], vlim_band[band]['max'])

                    ax.plot(time_vec, data_chunk, label='CHARGE',color='r')
                    ax.fill_between(time_vec, data_chunk+sem, data_chunk-sem, alpha=0.25, color='m')

                    ax.plot(time_vec, baseline, label='VS', color='b')
                    ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                    if fc_metric == 'MI':
                        _clusters = clusters.loc[pair, :].values
                    else:
                        _clusters = clusters.loc[band, pair, :].values
                        
                    ax.fill_between(time_vec, vlim_band[band]['min'], vlim_band[band]['max'], where=_clusters.astype('int'), alpha=0.3, color='r')

                    if stretch:
                        ax.vlines(stretch_point_ERP/2, ymin=vlim_band[band]['min'], ymax=vlim_band[band]['max'], colors='g')  
                    else:
                        ax.vlines(0, ymin=vlim_band[band]['min'], ymax=vlim_band[band]['max'], colors='g')  

                    fig.tight_layout()
                    plt.legend()

                    # plt.show()

                    #### save
                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'allpairs'))

                    if fc_metric == 'MI':

                        if stretch:

                            fig.savefig(f'stretch_{pair}.jpeg', dpi=150)

                        else:

                            fig.savefig(f'nostretch_{pair}.jpeg', dpi=150)

                    else:

                        if stretch:

                            fig.savefig(f'{band}_stretch_{pair}.jpeg', dpi=150)

                        else:

                            fig.savefig(f'{band}_nostretch_{pair}.jpeg', dpi=150)

                    fig.clf()
                    plt.close('all')
                    gc.collect()





def plot_allsujet_FC_mat():

    #stretch = True
    for stretch in [True, False]:

        #fc_metric = 'MI'
        for fc_metric in ['MI', 'ISPC', 'WPLI']:

            print(f'{fc_metric} PLOT stretch:{stretch}', flush=True)

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
            if stretch:
                fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
                clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_stretch.nc')
            else:
                fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
                clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS.nc')

            pairs_to_compute = fc_allsujet['pair'].values
            time_vec = fc_allsujet['time'].values
            phase_list = ['whole', 'inspi', 'expi']
            phase_vec = {'whole' : time_vec, 'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

            #band_i, band = 0, freq_band_fc_list
            for band_i, band in enumerate(freq_band_fc_list):

                #### stretch compute
                if stretch:

                    fc_mat = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    for phase_i, phase in enumerate(phase_list):

                        #pair_i, pair = 2, pairs_to_compute[2]
                        for pair_i, pair in enumerate(pairs_to_compute):

                            A, B = pair.split('-')
                            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                            if fc_metric == 'MI':
                                data_chunk_diff = fc_allsujet.loc[:, pair, 'CHARGE', phase_vec[phase]].mean('sujet').values - fc_allsujet.loc[:, pair, 'VS', phase_vec[phase]].mean('sujet').values
                                _clusters = clusters.loc[pair, phase_vec[phase]].values

                            else:
                                data_chunk_diff = fc_allsujet.loc[:, band, 'CHARGE', pair, phase_vec[phase]].mean('sujet').values - fc_allsujet.loc[:, band, 'VS', pair, phase_vec[phase]].mean('sujet').values
                                _clusters = clusters.loc[band,pair, phase_vec[phase]].values

                            if _clusters.sum() == 0:
                                continue

                            fc_val = data_chunk_diff[_clusters.astype('bool')].mean()

                            fc_mat[phase_i, A_i, B_i], fc_mat[phase_i, B_i, A_i] = fc_val, fc_val

                    #### plot

                    vlim = np.abs((fc_mat.min(), fc_mat.max())).max()

                    fig, axs = plt.subplots(ncols=len(phase_list), figsize=(12,5)) 

                    for phase_i, phase in enumerate(phase_list):

                        ax = axs[phase_i]

                        im = ax.imshow(fc_mat[phase_i, :, :], cmap='seismic', vmin=-vlim, vmax=vlim)
                        ax.set_xticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short, rotation=90)
                        ax.set_xlabel("Electrodes")

                        if phase_i == 0:
                            ax.set_yticks(ticks=np.arange(fc_mat.shape[1]), labels=chan_list_eeg_short)
                            ax.set_ylabel("Electrodes")

                        ax.set_title(phase)

                    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity Strength")

                    if fc_metric == 'MI':
                        plt.suptitle("MI FC")
                    else:
                        plt.suptitle(f'{fc_metric} {band} FC')

                #### nostretch comute
                else:  

                    fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                        if fc_metric == 'MI':
                            data_chunk_diff = fc_allsujet.loc[:, pair, 'CHARGE', time_vec].mean('sujet').values - fc_allsujet.loc[:, pair, 'VS', time_vec].mean('sujet').values
                            _clusters = clusters.loc[pair, :].values

                        else:
                            data_chunk_diff = fc_allsujet.loc[:, band, 'CHARGE', pair, time_vec].mean('sujet').values - fc_allsujet.loc[:, band, 'VS', pair, time_vec].mean('sujet').values
                            _clusters = clusters.loc[band,pair, :].values

                        if _clusters.sum() == 0:
                            continue

                        fc_val = data_chunk_diff[_clusters.astype('bool')].mean()

                        fc_mat[A_i, B_i], fc_mat[B_i, A_i] = fc_val, fc_val

                    #### plot

                    vlim = np.abs((fc_mat.min(), fc_mat.max())).max()

                    plt.matshow(fc_mat, cmap='seismic')
                    plt.colorbar(label='Connectivity Strength')
                    plt.clim(-vlim, vlim)
                    plt.xticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
                    plt.yticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short)
                    plt.xlabel("Electrodes")
                    plt.ylabel("Electrodes")
                    if fc_metric == 'MI':
                        plt.title("MI FC")
                    else:
                        plt.title(f'{fc_metric} {band} FC')
                    # plt.show()

                #### both save
                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                if fc_metric == 'MI':
                    if stretch:
                        plt.savefig(f'stretch_MI_FC.jpeg', dpi=150)
                    else:
                        plt.savefig(f'nostretch_MI_FC.jpeg', dpi=150)

                else:
                    if stretch:
                        plt.savefig(f'stretch_{fc_metric}_{band}_FC.jpeg', dpi=150)
                    else:
                        plt.savefig(f'nostretch_{fc_metric}_{band}_FC.jpeg', dpi=150)

                plt.close('all')
                gc.collect()

                def get_visu_data(time_window):

                    time_chunk_points = int(time_window * srate)
                    start_window = np.arange(0, time_vec.size, time_chunk_points)
                    start_window_sec = time_vec[start_window]
                    n_times = np.arange(start_window.size)
                    
                    cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #win_i, win_start = 15, start_window[15]
                    for win_i, win_start in enumerate(start_window):

                        if win_start == start_window[-1]:
                            continue
                        
                        win_stop = start_window[win_i+1]

                        _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                        _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                        #pair_i, pair = 2, pairs_to_compute[2]
                        for pair_i, pair in enumerate(pairs_to_compute):
                            A, B = pair.split('-')
                            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
                            cond_i, baseline_i = np.where(fc_allsujet['cond'].values == 'CHARGE')[0][0], np.where(fc_allsujet['cond'].values == 'VS')[0][0] 

                            if fc_metric == 'MI':
                                data_chunk_diff = fc_allsujet[:, pair_i, cond_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, pair_i, baseline_i, win_start:win_stop].mean('sujet').values
                                _clusters = clusters[pair_i, win_start:win_stop].values
                            else:
                                data_chunk_diff = fc_allsujet[:, band_i, cond_i, pair_i, win_start:win_stop].mean('sujet').values - fc_allsujet[:, band_i, baseline_i, pair_i, win_start:win_stop].mean('sujet').values
                                _clusters = clusters[band_i, pair_i, win_start:win_stop].values

                            fc_val = data_chunk_diff.mean()
                            _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                            
                            cluster_val = _clusters.sum() > 0
                            _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

                        data_chunk_wins[win_i, :,:] = _fc_mat
                        cluster_mask_wins[win_i, :,:] = _cluster_mat

                    return n_times, start_window, data_chunk_wins, cluster_mask_wins

                time_window = 0.1#in s
                n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data(time_window)
                vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

                if debug:

                    win_i = 15

                    plt.matshow(data_chunk_wins[win_i,:,:], cmap='seismic')

                    for i in range(chan_list_eeg_short.shape[0]):
                        for j in range(chan_list_eeg_short.shape[0]):
                            if cluster_mask_wins[win_i, i, j]:
                                plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))

                    plt.colorbar(label='Connectivity Strength')
                    plt.clim(-vlim, vlim)
                    plt.xticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
                    plt.yticks(ticks=np.arange(fc_mat.shape[0]), labels=chan_list_eeg_short)
                    plt.xlabel("Electrodes")
                    plt.ylabel("Electrodes")
                    if fc_metric == 'MI':
                        plt.title(f"MI FC start{np.round(time_vec[start_window[win_i]], 2)}")
                    else:
                        plt.title(f"{fc_metric} {band} FC start{np.round(time_vec[start_window[win_i]], 2)}")
                    plt.show()

                # Create topoplot frames
                fig, ax = plt.subplots()
                cax = ax.matshow(np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short))), cmap='seismic', vmin=-vlim, vmax=vlim)
                colorbar = plt.colorbar(cax, ax=ax)

                def update(frame):
                    
                    ax.clear()
                    ax.set_title(np.round(time_vec[start_window[frame]], 5))
                    
                    ax.matshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

                    for i in range(chan_list_eeg_short.shape[0]):
                        for j in range(chan_list_eeg_short.shape[0]):
                            if cluster_mask_wins[frame, i, j]:
                                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                                edgecolor='green', facecolor='none', lw=4)
                                ax.add_patch(rect)

                    ax.set_xticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short, rotation=90)
                    ax.set_yticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short)
                    ax.set_xlabel("Electrodes")
                    ax.set_ylabel("Electrodes")
                    ax.set_title(f"MI FC start{np.round(time_vec[start_window[frame]], 2)}")

                    return [ax]

                # Animation
                ani = FuncAnimation(fig, update, frames=n_times, interval=1000)
                # plt.show()

                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                
                if stretch:
                    if fc_metric == 'MI':
                        ani.save(f"stretch_{fc_metric}_FC_mat_animation_allsujet.gif", writer="pillow")
                    else:
                        ani.save(f"stretch_{fc_metric}_{band}_FC_mat_animation_allsujet.gif", writer="pillow")  
                else:
                    if fc_metric == 'MI':
                        ani.save(f"nostretch_{fc_metric}_FC_mat_animation_allsujet.gif", writer="pillow")
                    else:
                        ani.save(f"nostretch_{fc_metric}_{band}_FC_mat_animation_allsujet.gif", writer="pillow")  






def plot_allsujet_FC_graph_nostretch():

    stretch=False

    #fc_metric = 'ISPC'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch:{stretch}', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        if stretch:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
            clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_stretch.nc')
        else:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
            clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS.nc')

        pairs_to_compute = fc_allsujet['pair'].values
        time_vec = fc_allsujet['time'].values

        for band_i, band in enumerate(freq_band_fc_list):

            #### generate matrix
            fc_mat_cond = np.zeros((len(sujet_list), len(cond_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

            for sujet_i, sujet in enumerate(sujet_list):

                for cond_i, cond in enumerate(cond_list):
                
                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                        if fc_metric == 'MI':
                            fc_val = fc_allsujet.loc[sujet, pair, cond, time_vec].values.mean()
                        else:
                            fc_val = fc_allsujet.loc[sujet, band, cond, pair, time_vec].values.mean()

                        fc_mat_cond[sujet_i, cond_i, A_i, B_i], fc_mat_cond[sujet_i, cond_i, B_i, A_i] = fc_val, fc_val

            fc_mat_cond = xr.DataArray(data=fc_mat_cond, dims=['sujet', 'cond', 'chanA', 'chanB'], coords=[sujet_list, cond_list,chan_list_eeg_short, chan_list_eeg_short])

            if debug:

                plt.matshow(fc_mat_cond[0,:,:])
                plt.colorbar(label='Connectivity Strength')
                plt.xticks(ticks=np.arange(fc_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short, rotation=90)
                plt.yticks(ticks=np.arange(fc_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short)
                plt.xlabel("Electrodes")
                plt.ylabel("Electrodes")
                plt.title(f"{fc_metric} FC")
                plt.show()

            #mat = fc_mat_cond[0]
            def thresh_fc_mat(mat, percentile_graph_metric=50):

                mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                
                if debug:
                    np.sum(mat_values > np.percentile(mat_values, 90))

                    count, bin, fig = plt.hist(mat_values)
                    plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 75), ymin=count.min(), ymax=count.max(), color='r')
                    plt.show()

                #### apply thresh
                for chan_i in range(mat.shape[0]):
                    mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

                #### verify that the graph is fully connected
                chan_i_to_remove = []
                for chan_i in range(mat.shape[0]):
                    if np.sum(mat[chan_i,:]) == 0:
                        chan_i_to_remove.append(chan_i)

                mat_i_mask = [i for i in range(mat.shape[0]) if i not in chan_i_to_remove]

                if len(chan_i_to_remove) != 0:
                    for row in range(2):
                        if row == 0:
                            mat = mat[mat_i_mask,:]
                        elif row == 1:
                            mat = mat[:,mat_i_mask]

                if debug:
                    plt.matshow(mat)
                    plt.show()

                return mat
            
            df_graph_metrics = pd.DataFrame()

            for sujet in sujet_list:

                for cond in cond_list:

                    # Create a graph from the adjacency matrix
                    mat = thresh_fc_mat(fc_mat_cond.loc[sujet,cond,:,:].values)
                    graph = nx.from_numpy_array(mat)
                    nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

                    if debug:
                        # Plot the graph
                        plt.figure(figsize=(10, 8))
                        pos = nx.spring_layout(graph)  # Layout for visualization
                        nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
                        plt.title("Graph Visualization")
                        plt.show()


                    #### metrics
                    degree = nx.degree_centrality(graph)
                    # Formula: degree_centrality(v) = degree(v) / (n - 1), where n is the number of nodes

                    betweenness = nx.betweenness_centrality(graph)
                    # Formula: betweenness_centrality(v) = sum of (shortest paths through v / total shortest paths)

                    closeness = nx.closeness_centrality(graph)
                    # Formula: closeness_centrality(v) = 1 / (sum of shortest path distances from v to all other nodes)

                    clustering_coeff = nx.clustering(graph)

                    local_efficiency = nx.local_efficiency(graph)

                    # Hubness (using HITS algorithm)
                    hubs, authorities = nx.hits(graph)
                    # Formula: HITS hub score: importance of a node as a hub, based on linking to authorities

                    for chan in chan_list_eeg:

                        try:
                            _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'chan' : [chan], 'degree' : [degree[chan]], 'betweenness' : [betweenness[chan]], 
                                    'closeness' : [closeness[chan]], 'hubs' : [hubs[chan]],
                                    'clustering_coeff' : [clustering_coeff[chan]], 'local_efficiency' : [local_efficiency]})

                            df_graph_metrics = pd.concat((df_graph_metrics, _df))
                        except:
                            pass

            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

            for metric in ['degree', 'betweenness', 'closeness', 'hubs', 'clustering_coeff', 'local_efficiency']:

                g = sns.catplot(
                    data=df_graph_metrics, kind="bar",
                    x="chan", y=metric, hue="cond",
                    alpha=.6, height=6)
                # plt.show()

                if stretch:
                    if fc_metric == 'MI':
                        plt.savefig(f"stretch_{fc_metric}_graph_{metric}.png")
                    else:
                        plt.savefig(f"stretch_{fc_metric}_{band}_graph_{metric}.png")
                else:
                    if fc_metric == 'MI':
                        plt.savefig(f"nostretch_{fc_metric}_graph_{metric}.png")
                    else:
                        plt.savefig(f"nostretch_{fc_metric}_{band}_graph_{metric}.png")

                plt.close('all')






def plot_allsujet_FC_graph_stretch():

    stretch=True

    #fc_metric = 'MI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch:{stretch}', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
        if stretch:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet_stretch.nc')
            clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_stretch.nc')
        else:
            fc_allsujet = xr.open_dataarray(f'{fc_metric}_allsujet.nc')
            clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS.nc')

        pairs_to_compute = fc_allsujet['pair'].values
        time_vec = fc_allsujet['time'].values
        phase_list = ['whole', 'inspi', 'expi']
        phase_vec = {'whole' : time_vec, 'inspi' : np.arange(stretch_point_ERP/2).astype('int'), 'expi' : (np.arange(stretch_point_ERP/2)+stretch_point_ERP/2).astype('int')}

        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            #### generate matrix
            fc_mat_cond = np.zeros((len(sujet_list), len(cond_list), len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

            for sujet_i, sujet in enumerate(sujet_list):

                for cond_i, cond in enumerate(cond_list):

                    for phase_i, phase in enumerate(phase_list):
                
                        #pair_i, pair = 2, pairs_to_compute[2]
                        for pair_i, pair in enumerate(pairs_to_compute):

                            A, B = pair.split('-')
                            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                            if fc_metric == 'MI':
                                fc_val = fc_allsujet.loc[sujet, pair, cond, phase_vec[phase]].values.mean()
                            else:
                                fc_val = fc_allsujet.loc[sujet, band, cond, pair, phase_vec[phase]].values.mean()

                            fc_mat_cond[sujet_i, cond_i, phase_i, A_i, B_i], fc_mat_cond[sujet_i, cond_i, phase_i, B_i, A_i] = fc_val, fc_val

            fc_mat_cond = xr.DataArray(data=fc_mat_cond, dims=['sujet', 'cond', 'phase', 'chanA', 'chanB'], coords=[sujet_list, cond_list, phase_list, chan_list_eeg_short, chan_list_eeg_short])

            if debug:

                plt.matshow(fc_mat_cond[0,:,:])
                plt.colorbar(label='Connectivity Strength')
                plt.xticks(ticks=np.arange(fc_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short, rotation=90)
                plt.yticks(ticks=np.arange(fc_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short)
                plt.xlabel("Electrodes")
                plt.ylabel("Electrodes")
                plt.title(f"{fc_metric} FC")
                plt.show()

            #mat = fc_mat_cond[0]
            def thresh_fc_mat(mat, percentile_graph_metric=50):

                mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                
                if debug:
                    np.sum(mat_values > np.percentile(mat_values, 90))

                    count, bin, fig = plt.hist(mat_values)
                    plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                    plt.vlines(np.percentile(mat_values, 75), ymin=count.min(), ymax=count.max(), color='r')
                    plt.show()

                #### apply thresh
                for chan_i in range(mat.shape[0]):
                    mat[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

                #### verify that the graph is fully connected
                chan_i_to_remove = []
                for chan_i in range(mat.shape[0]):
                    if np.sum(mat[chan_i,:]) == 0:
                        chan_i_to_remove.append(chan_i)

                mat_i_mask = [i for i in range(mat.shape[0]) if i not in chan_i_to_remove]

                if len(chan_i_to_remove) != 0:
                    for row in range(2):
                        if row == 0:
                            mat = mat[mat_i_mask,:]
                        elif row == 1:
                            mat = mat[:,mat_i_mask]

                if debug:
                    plt.matshow(mat)
                    plt.show()

                return mat
            
            df_graph_metrics = pd.DataFrame()

            for sujet in sujet_list:

                for cond in cond_list:

                    for phase in phase_list:

                        # Create a graph from the adjacency matrix
                        mat = thresh_fc_mat(fc_mat_cond.loc[sujet,cond,phase,:,:].values)
                        graph = nx.from_numpy_array(mat)
                        nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

                        if debug:
                            # Plot the graph
                            plt.figure(figsize=(10, 8))
                            pos = nx.spring_layout(graph)  # Layout for visualization
                            nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
                            plt.title("Graph Visualization")
                            plt.show()


                        #### metrics
                        degree = nx.degree_centrality(graph)
                        # Formula: degree_centrality(v) = degree(v) / (n - 1), where n is the number of nodes

                        betweenness = nx.betweenness_centrality(graph)
                        # Formula: betweenness_centrality(v) = sum of (shortest paths through v / total shortest paths)

                        closeness = nx.closeness_centrality(graph)
                        # Formula: closeness_centrality(v) = 1 / (sum of shortest path distances from v to all other nodes)

                        clustering_coeff = nx.clustering(graph)

                        local_efficiency = nx.local_efficiency(graph)

                        # Hubness (using HITS algorithm)
                        hubs, authorities = nx.hits(graph)
                        # Formula: HITS hub score: importance of a node as a hub, based on linking to authorities

                        for chan in chan_list_eeg:

                            try:
                                _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'phase' : [phase], 'chan' : [chan], 'degree' : [degree[chan]], 'betweenness' : [betweenness[chan]], 
                                        'closeness' : [closeness[chan]], 'hubs' : [hubs[chan]],
                                        'clustering_coeff' : [clustering_coeff[chan]], 'local_efficiency' : [local_efficiency]})

                                df_graph_metrics = pd.concat((df_graph_metrics, _df))
                            except:
                                pass

            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

            #metric = 'degree'
            for metric in ['degree', 'betweenness', 'closeness', 'hubs', 'clustering_coeff', 'local_efficiency']:

                fig, axs = plt.subplots(ncols=len(phase_list), figsize=(15, 5), sharey=True)

                for phase_i, phase in enumerate(phase_list):

                    ax = axs[phase_i]
                    sns.barplot(data=df_graph_metrics, x="chan", y=metric, hue="cond", alpha=0.6, ax=ax)
                    ax.set_title(phase) 
                    ax.set_xlabel("Channel")
                    if phase_i == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")

                if stretch:
                    if fc_metric == 'MI':
                        plt.savefig(f"stretch_{fc_metric}_graph_{metric}.png")
                    else:
                        plt.savefig(f"stretch_{fc_metric}_{band}_graph_{metric}.png")
                else:
                    if fc_metric == 'MI':
                        plt.savefig(f"nostretch_{fc_metric}_graph_{metric}.png")
                    else:
                        plt.savefig(f"nostretch_{fc_metric}_{band}_graph_{metric}.png")

                plt.close("all")












################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    plot_allsujet_FC_chunk()
    plot_allsujet_FC_mat()
    plot_allsujet_FC_graph_nostretch()
    plot_allsujet_FC_graph_stretch()




