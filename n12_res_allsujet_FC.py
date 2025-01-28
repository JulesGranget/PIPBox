
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
######## PLOT ########
################################

def plot_allsujet_MI_chunk():

    #stretch = False
    for stretch in [True, False]:

        print(f'MI PLOT stretch:{stretch}', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        if stretch:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs_stretch.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS_stretch.nc')
        else:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS.nc')

        pairs_to_compute = MI_allsujet['pair'].values
        time_vec = MI_allsujet['time'].values

        ######## IDENTIFY MIN MAX ########

        #### identify min max for allsujet

        vlim = np.array([])

        #pair_i, pair = 0, pairs_to_compute[0]
        for pair_i, pair in enumerate(pairs_to_compute):

            #cond_i, cond = 0, cond_list[0]
            for cond_i, cond in enumerate(cond_list):

                data_chunk = MI_allsujet.loc[:, pair, cond, :].mean('sujet').values
                sem = MI_allsujet.loc[:, pair, cond, :].std('sujet').values / np.sqrt(MI_allsujet.loc[:, pair, cond, :].shape[0])
                data_chunk_up, data_chunk_down = data_chunk + sem, data_chunk - sem
                vlim = np.concatenate([vlim, data_chunk_up, data_chunk_down])

        vlim = {'min' : vlim.min(), 'max' : vlim.max()}

        if debug:

            for pair in pairs_to_compute:

                plt.plot(MI_allsujet.loc[:, 'CHARGE', pair, :].mean('sujet'))
            
            plt.show()

        n_sujet = MI_allsujet['sujet'].shape[0]

        #pair_i, pair = 0, pairs_to_compute[0]
        for pair_i, pair in enumerate(pairs_to_compute):

            fig, ax = plt.subplots()

            fig.set_figheight(5)
            fig.set_figwidth(8)

            if stretch:
                plt.suptitle(f'stretch {pair} nsujet:{n_sujet}')
            else:
                plt.suptitle(f'{pair} nsujet:{n_sujet}')

            data_stretch = MI_allsujet.loc[:, pair, 'CHARGE', :].mean('sujet').values
            sem = MI_allsujet.loc[:, pair, 'CHARGE', :].std('sujet').values / np.sqrt(MI_allsujet.loc[:, pair, 'CHARGE', :].shape[0])
            baseline = MI_allsujet.loc[:, pair, 'VS', :].mean('sujet').values
            sem_baseline = MI_allsujet.loc[:, pair, 'VS', :].std('sujet').values / np.sqrt(MI_allsujet.loc[:, pair, 'VS', :].shape[0])

            ax.set_ylim(vlim['min'], vlim['max'])

            ax.plot(time_vec, data_stretch, label='CHARGE',color='r')
            ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

            ax.plot(time_vec, baseline, label='VS', color='b')
            ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

            _clusters = clusters.loc[pair, :].values
            ax.fill_between(time_vec, vlim['min'], vlim['max'], where=_clusters.astype('int'), alpha=0.3, color='r')

            if stretch:
                ax.vlines(stretch_point_ERP/2, ymin=vlim['min'], ymax=vlim['max'], colors='g')  
            else:
                ax.vlines(0, ymin=vlim['min'], ymax=vlim['max'], colors='g')  

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'FC', 'MI', 'allpairs'))

            if stretch:

                fig.savefig(f'stretch_{pair}.jpeg', dpi=150)

            else:

                fig.savefig(f'nostretch_{pair}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()



def plot_allsujet_MI_mat():

    #stretch = False
    for stretch in [True, False]:

        print(f'MI PLOT stretch:{stretch}', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        if stretch:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs_stretch.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS_stretch.nc')
        else:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS.nc')

        pairs_to_compute = MI_allsujet['pair'].values
        time_vec = MI_allsujet['time'].values

        MI_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

        if stretch:
            time_vec_stats = time_vec[time_vec <= 0]
        else:
            time_vec_stats = time_vec

        #pair_i, pair = 2, pairs_to_compute[2]
        for pair_i, pair in enumerate(pairs_to_compute):
            A, B = pair.split('-')
            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

            data_chunk_diff = MI_allsujet.loc[:, pair, 'CHARGE', time_vec_stats].mean('sujet').values - MI_allsujet.loc[:, pair, 'VS', time_vec_stats].mean('sujet').values
            _clusters = clusters.loc[pair, :].values

            if _clusters.sum() == 0:
                continue

            fc_val = data_chunk_diff[_clusters.astype('bool')].mean()

            MI_mat[A_i, B_i], MI_mat[B_i, A_i] = fc_val, fc_val

        #### plot

        vlim = np.abs((MI_mat.min(), MI_mat.max())).max()

        plt.matshow(MI_mat, cmap='seismic')
        plt.colorbar(label='Connectivity Strength')
        plt.clim(-vlim, vlim)
        plt.xticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
        plt.yticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short)
        plt.xlabel("Electrodes")
        plt.ylabel("Electrodes")
        plt.title("MI FC")
        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'FC', 'MI', 'matplot'))

        if stretch:

            plt.savefig(f'stretch_MI_FC.jpeg', dpi=150)

        else:

            plt.savefig(f'nostretch_MI_FC.jpeg', dpi=150)

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

            _MI_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
            _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

            #pair_i, pair = 2, pairs_to_compute[2]
            for pair_i, pair in enumerate(pairs_to_compute):
                A, B = pair.split('-')
                A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]
                cond_i, baseline_i = np.where(MI_allsujet['cond'].values == 'CHARGE')[0][0], np.where(MI_allsujet['cond'].values == 'VS')[0][0] 

                data_chunk_diff = MI_allsujet[:, pair_i, cond_i, win_start:win_stop].mean('sujet').values - MI_allsujet[:, pair_i, baseline_i, win_start:win_stop].mean('sujet').values
                fc_val = data_chunk_diff.mean()
                _MI_mat[A_i, B_i], _MI_mat[B_i, A_i] = fc_val, fc_val
                
                _clusters = clusters[pair_i, win_start:win_stop].values
                cluster_val = _clusters.sum() > 0
                _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

            data_chunk_wins[win_i, :,:] = _MI_mat
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
        plt.xticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
        plt.yticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short)
        plt.xlabel("Electrodes")
        plt.ylabel("Electrodes")
        plt.title(f"MI FC start{np.round(time_vec[start_window[win_i]], 2)}")
        plt.show()

    # Create topoplot frames
    fig, ax = plt.subplots()
    cax = ax.matshow(np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short))), cmap='seismic', vmin=-vlim, vmax=vlim)
    colorbar = plt.colorbar(cax, ax=ax)

    def update(frame):
        global cbar
        ax.clear()
        ax.set_title(np.round(time_vec[start_window[frame]], 5))
        
        ax.matshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

        for i in range(chan_list_eeg_short.shape[0]):
            for j in range(chan_list_eeg_short.shape[0]):
                if cluster_mask_wins[frame, i, j]:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                     edgecolor='green', facecolor='none', lw=2)
                    ax.add_patch(rect)

        ax.set_xticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short, rotation=90)
        ax.set_yticks(ticks=np.arange(MI_mat.shape[0]), labels=chan_list_eeg_short)
        ax.set_xlabel("Electrodes")
        ax.set_ylabel("Electrodes")
        ax.set_title(f"MI FC start{np.round(time_vec[start_window[frame]], 2)}")

        return [ax, cbar]

    # Animation
    ani = FuncAnimation(fig, update, frames=n_times, interval=1000)
    # plt.show()

    os.chdir(os.path.join(path_results, 'FC', 'MI', 'matplot'))
    ani.save("FC_MI_mat_animation_allsujet.gif", writer="pillow")  










def plot_allsujet_MI_graph():

    #stretch = False
    for stretch in [True, False]:

        print(f'MI PLOT stretch:{stretch}', flush=True)

        os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
        if stretch:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs_stretch.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS_stretch.nc')
        else:
            MI_allsujet = xr.open_dataarray('allsujet_MI_allpairs.nc')
            clusters = xr.open_dataarray('allsujet_MI_STATS.nc')

        pairs_to_compute = MI_allsujet['pair'].values
        time_vec = MI_allsujet['time'].values

        if not stretch:
            time_vec_stats = time_vec[time_vec <= 0]
        else:
            time_vec_stats = time_vec

        #### generate matrix
        MI_mat_cond = np.zeros((len(sujet_list), len(cond_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

        for sujet_i, sujet in enumerate(sujet_list):

            for cond_i, cond in enumerate(cond_list):
            
                #pair_i, pair = 2, pairs_to_compute[2]
                for pair_i, pair in enumerate(pairs_to_compute):

                    A, B = pair.split('-')
                    A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                    fc_val = MI_allsujet.loc[sujet, pair, cond, time_vec_stats].values.mean()

                    MI_mat_cond[sujet_i, cond_i, A_i, B_i], MI_mat_cond[sujet_i, cond_i, B_i, A_i] = fc_val, fc_val

        MI_mat_cond = xr.DataArray(data=MI_mat_cond, dims=['sujet', 'cond', 'chanA', 'chanB'], coords=[sujet_list, cond_list,chan_list_eeg_short, chan_list_eeg_short])

        if debug:

            plt.matshow(MI_mat_cond[0,:,:])
            plt.colorbar(label='Connectivity Strength')
            plt.xticks(ticks=np.arange(MI_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short, rotation=90)
            plt.yticks(ticks=np.arange(MI_mat_cond[0,:,:].shape[0]), labels=chan_list_eeg_short)
            plt.xlabel("Electrodes")
            plt.ylabel("Electrodes")
            plt.title(f"MI FC")
            plt.show()

        #mat = MI_mat_cond[0]
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
                mat = thresh_fc_mat(MI_mat_cond.loc[sujet,cond,:,:].values)
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

        os.chdir(os.path.join(path_results, 'FC', 'MI', 'graph'))
        for metric in ['degree', 'betweenness', 'closeness', 'hubs', 'clustering_coeff', 'local_efficiency']:

            g = sns.catplot(
                data=df_graph_metrics, kind="bar",
                x="chan", y=metric, hue="cond",
                alpha=.6, height=6)
            # plt.show()

            plt.savefig(f"graph_{metric}.png")



################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    plot_allsujet_MI_chunk()
    plot_allsujet_MI_mat()
    plot_allsujet_MI_graph()




