


from n00_config_params import *
from n00bis_config_analysis_functions import *


debug = False




def test_FC_metrics():
    


    def generate_synchronized_signals(duration_tot=1200, num_windows=50, window_dur=5, noise_coeff=1, freq_sync=10, fs=500, nshift=1000, amp_coeff=2):
            
        total_samples = duration_tot * fs  # Total signal length in samples
        t = np.arange(total_samples) / fs  # Time vector

        # Initialize signals with random noise
        signal1 = np.random.randn(total_samples) * noise_coeff
        signal2 = np.random.randn(total_samples) * noise_coeff

        # Create oscillatory activity at the common frequency for the entire duration
        osc_total = np.sin(2 * np.pi * freq_sync * t)

        # Add the oscillatory component to both signals (not synchronized by default)
        signal1 += osc_total

        shift_indices = np.random.randint(low=0, high=signal1.size, size=nshift)
        shift_indices = sorted(shift_indices)  # Get random non-overlapping windows

        for i in shift_indices:
            signal2[i:] *= -1

        if debug:
            plt.plot(signal1)
            plt.plot(signal2)
            plt.show()

        # Ensure non-overlapping windows with at least one window in between
        available_indices = np.arange(window_dur * fs, total_samples - window_dur * fs, window_dur * fs)  # Possible start points
        available_indices = available_indices[np.arange(0,available_indices.size,2)]

        if available_indices.size < num_windows:
            raise ValueError('sig too small for num_windows')
        else:
            sel_rand_win = np.random.choice(np.arange(available_indices.size), size=num_windows, replace=False)
            sync_windows = available_indices[sel_rand_win]

        # Apply synchronization in selected windows
        for start in sync_windows:
            end = start + window_dur * fs  # 5 sec duration
            osc = np.sin(2 * np.pi * freq_sync * np.arange(0, window_dur, 1/fs))

            # Overwrite the signal in the synchronization window
            signal1[start:end] = osc * amp_coeff + np.random.randn(osc.size) * noise_coeff
            signal2[start:end] = osc * amp_coeff + np.random.randn(osc.size) * noise_coeff

        return signal1, signal2, sync_windows



    
    def morlet_wavelet_transform(signal_data, fs=500, freq_sync=6, num_cycles=7):
        
        # Define time window for the wavelet
        t = np.arange(-5, 5, 1/fs)  # Time vector
        sigma_t = num_cycles / (2 * np.pi * freq_sync)  # Standard deviation in time

        # Manually create the Morlet wavelet
        wavelet = np.exp(2j * np.pi * freq_sync * t) * np.exp(-t**2 / (2 * sigma_t**2))

        # Convolve wavelet with signal using FFT for speed
        analytic_signal = scipy.signal.fftconvolve(signal_data, wavelet, mode='same')

        return analytic_signal

    # Generate signals
    duration_tot=1000
    num_windows=50
    window_dur=5
    fs=100
    nshift=1000
    freq_sync=8
    amp_coeff=2

    signal1, signal2, sync_windows = generate_synchronized_signals(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                   fs=fs, nshift=nshift, freq_sync=freq_sync, amp_coeff=amp_coeff)

    if debug:

        plt.plot(signal1)
        plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
        plt.plot(signal2)
        plt.show()

    # Extract analytic signal using Morlet wavelet
    x = morlet_wavelet_transform(signal1, fs=fs, freq_sync=freq_sync, num_cycles=7)
    y = morlet_wavelet_transform(signal2, fs=fs, freq_sync=freq_sync, num_cycles=7)

    if debug:

        x_plot = np.abs(x)**2
        plt.plot(x_plot)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

    # conv metrics
    # win_conv = 1*fs
    ncycle_FC = 10
    win_conv = int(ncycle_FC/freq_sync*fs)

    pre_win, post_win = 4, 4
    res_win_size_fc = np.zeros((3, np.arange(2,30,4).size, (window_dur+pre_win+post_win)*fs))

    for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,4)):

        win_conv = int(ncycle_FC/freq_sync*fs)

        signal1_pad = np.pad(signal1, int(win_conv/2), mode='reflect')
        signal2_pad = np.pad(signal2, int(win_conv/2), mode='reflect')

        x_pad = np.pad(x, int(win_conv/2), mode='reflect')
        y_pad = np.pad(y, int(win_conv/2), mode='reflect')

        # conv
        MI_conv = np.array([get_MI_2sig(signal1_pad[i:i+win_conv], signal2_pad[i:i+win_conv]) for i in range(int(signal1_pad.size-win_conv))])

        if debug:

            x_plot = MI_conv
            plt.plot(x_plot)
            plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
            plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
            plt.show()

        ISPC_conv = np.array([get_ISPC_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])

        if debug:

            x_plot = ISPC_conv
            plt.plot(x_plot)
            plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
            plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
            plt.show()

        WPLI_conv = np.array([get_WPLI_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])

        if debug:

            x_plot = WPLI_conv
            plt.plot(x_plot)
            plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
            plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
            plt.show()

        epochs_MI = []
        epochs_ISPC = []
        epochs_WPLI = []
        
        for win_i, win_time in enumerate(sync_windows):

            start, stop = win_time-pre_win*fs, win_time+window_dur*fs+post_win*fs
            if start < 0 or stop > duration_tot*fs:
                continue
            epochs_ISPC.append(ISPC_conv[start:stop])
            epochs_MI.append(MI_conv[start:stop])
            epochs_WPLI.append(WPLI_conv[start:stop])

        epochs_ISPC = np.array(epochs_ISPC)
        epochs_MI = np.array(epochs_MI)
        epochs_WPLI = np.array(epochs_WPLI)

        res_win_size_fc[0,ncycle_FC_i,:], res_win_size_fc[1,ncycle_FC_i,:], res_win_size_fc[2,ncycle_FC_i,:] = np.median(epochs_ISPC, axis=0), np.median(epochs_MI, axis=0), np.median(epochs_WPLI, axis=0)

        #### plot for one ncycle
        os.chdir(os.path.join(path_results, 'FC'))

        plt.pcolormesh(epochs_ISPC)
        plt.title(f'ISPC_ncycle:{ncycle_FC}')
        plt.savefig(f'ISPC_raster_ncycle{ncycle_FC}.jpg')
        # plt.show()
        plt.close('all')

        plt.pcolormesh(epochs_MI)
        plt.title(f'MI_ncycle:{ncycle_FC}')
        plt.savefig(f'MI_raster_ncycle{ncycle_FC}.jpg')
        # plt.show()
        plt.close('all')

        plt.pcolormesh(epochs_WPLI)
        plt.title(f'WPLI_ncycle:{ncycle_FC}')
        plt.savefig(f'WPLI_raster_ncycle{ncycle_FC}.jpg')
        # plt.show()
        plt.close('all')

        plt.plot(scipy.stats.zscore(np.median(epochs_ISPC, axis=0)), label='ISPC')
        plt.plot(scipy.stats.zscore(np.median(epochs_MI, axis=0)), label='MI')
        plt.plot(scipy.stats.zscore(np.median(epochs_WPLI, axis=0)), label='WPLI')
        plt.legend()
        plt.savefig(f'median_FC_comparison_ncycle{ncycle_FC}.jpg')
        # plt.show()
        plt.close('all')

    #### plot for all ncycle
    os.chdir(os.path.join(path_results, 'FC'))
    
    for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
        plt.plot(res_win_size_fc[0,ncycle_FC_i,:], label=ncycle_FC)
    plt.legend()
    plt.title('ISPC')
    plt.savefig(f'median_ISPC_comparison_ncycle.jpg')
    # plt.show()
    plt.close('all')

    for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
        plt.plot(res_win_size_fc[1,ncycle_FC_i,:], label=ncycle_FC)
    plt.legend()
    plt.title('MI')
    plt.savefig(f'median_MI_comparison_ncycle.jpg')
    # plt.show()
    plt.close('all')

    for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
        plt.plot(res_win_size_fc[2,ncycle_FC_i,:], label=ncycle_FC)
    plt.legend()
    plt.title('WPLI')
    plt.savefig(f'median_WPLI_comparison_ncycle.jpg')
    # plt.show()
    plt.close('all')
    
    #### noise evaluation
    noise_vec = np.arange(1,5,0.5)
    ncycle_FC = 10
    win_conv = int(ncycle_FC/freq_sync*fs)
    pre_win, post_win = 4, 4
    res_noise_coeff_fc = np.zeros((3, noise_vec.size, (window_dur+pre_win+post_win)*fs))

    for noice_coeff_i, noise_coeff in enumerate(noise_vec):

        print(noise_coeff)

        signal1, signal2, sync_windows = generate_synchronized_signals(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                    fs=fs, nshift=nshift, freq_sync=freq_sync, amp_coeff=amp_coeff, noise_coeff=noise_coeff)
        
        if debug:

            plt.plot(signal1)
            plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
            plt.vlines((np.array(sync_windows)+window_dur*fs), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
            plt.plot(signal2)
            plt.show()
        
        signal1_pad = np.pad(signal1, int(win_conv/2), mode='reflect')
        signal2_pad = np.pad(signal2, int(win_conv/2), mode='reflect')

        x_pad = np.pad(x, int(win_conv/2), mode='reflect')
        y_pad = np.pad(y, int(win_conv/2), mode='reflect')

        # conv
        MI_conv = np.array([get_MI_2sig(signal1_pad[i:i+win_conv], signal2_pad[i:i+win_conv]) for i in range(int(signal1_pad.size-win_conv))])
        ISPC_conv = np.array([get_ISPC_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])
        WPLI_conv = np.array([get_WPLI_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])

        epochs_MI = []
        epochs_ISPC = []
        epochs_WPLI = []
        
        for win_i, win_time in enumerate(sync_windows):

            start, stop = win_time-pre_win*fs, win_time+window_dur*fs+post_win*fs
            if start < 0 or stop > duration_tot*fs:
                continue
            epochs_ISPC.append(ISPC_conv[start:stop])
            epochs_MI.append(MI_conv[start:stop])
            epochs_WPLI.append(WPLI_conv[start:stop])

        epochs_ISPC = np.array(epochs_ISPC)
        epochs_MI = np.array(epochs_MI)
        epochs_WPLI = np.array(epochs_WPLI)

        res_noise_coeff_fc[0,noice_coeff_i,:], res_noise_coeff_fc[1,noice_coeff_i,:], res_noise_coeff_fc[2,noice_coeff_i,:] = np.median(epochs_ISPC, axis=0), np.median(epochs_MI, axis=0), np.median(epochs_WPLI, axis=0)

    #### plot for all ncycle
    os.chdir(os.path.join(path_results, 'FC'))
    
    for noice_coeff_i, noise_coeff in enumerate(noise_vec):
        plt.plot(res_noise_coeff_fc[0,noice_coeff_i,:], label=noise_coeff)
    plt.legend()
    plt.title('ISPC')
    plt.savefig(f'median_ISPC_comparison_noise_coeff.jpg')
    # plt.show()
    plt.close('all')

    for noice_coeff_i, noise_coeff in enumerate(noise_vec):
        plt.plot(res_noise_coeff_fc[1,noice_coeff_i,:], label=noise_coeff)
    plt.legend()
    plt.title('MI')
    plt.savefig(f'median_MI_comparison_noise_coeff.jpg')
    # plt.show()
    plt.close('all')

    for noice_coeff_i, noise_coeff in enumerate(noise_vec):
        plt.plot(res_noise_coeff_fc[2,noice_coeff_i,:], label=noise_coeff)
    plt.legend()
    plt.title('WPLI')
    plt.savefig(f'median_WPLI_comparison_noise_coeff.jpg')
    # plt.show()
    plt.close('all')
    
