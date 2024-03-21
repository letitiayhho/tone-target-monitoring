def compute_power_dB(evokeds):
    poststim = evokeds.compute_psd(tmin = 0., tmax = 0.2)
    baseline = evokeds.compute_psd(tmin = -0.2, tmax = 0.)
    power = 10 * np.log10(poststim.get_data() / baseline.get_data())
    power = np.squeeze(power)
    freqs = poststim.freqs
    return freqs, power

def read_epochs(sub, desc):
    '''
    reads and concatenates epochs across runs
    '''
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    run = lambda f: int(re.findall('run-(\w+)_', f)[0])
    fnames = layout.get(
        return_type = 'filename',
        subject = sub, 
        desc = desc
        )
    print(fnames)
    fnames.sort(key = run)
    epochs_all = [mne.read_epochs(f) for f in fnames]
    epochs = mne.concatenate_epochs(epochs_all)
    epochs = epochs.pick('eeg')
    return epochs

def plot_psd_dB(freqs, power, stim_freq):
    plt.plot(freqs, power, label = str(stim_freq))
    plt.xlabel('frequency')
    plt.ylabel('dB')
    plt.xlim(0, 600)
    plt.ylim(0, 20)
    plt.axvline(stim_freq, linestyle = '--', color = 'grey')
    plt.axvline(stim_freq*2, linestyle = 'dotted', color = 'grey')
    plt.legend(title = 'Stim frequency (Hz)')
    
def compute_power_fft(fs, evokeds):
    x = evokeds.get_data()
    x = x.flatten()
    time_step = 1 / fs
    freqs = np.fft.fftfreq(x.size, time_step)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(x))**2
    return freqs, ps, idx

def plot_psd_fft(freqs, ps, idx, stim_freq):
    plt.plot(freqs[idx], ps[idx], label = str(stim_freq))
    plt.legend(title = 'Stim frequency (Hz)')
    plt.xlabel("Hz")
    plt.ylabel("PSD (V^2/Hz)")
    plt.xlim(0, 600)
    plt.axvline(stim_freq, linestyle = '--', color = 'grey')
    plt.axvline(stim_freq*2, linestyle = 'dotted', color = 'grey')