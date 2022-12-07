from scipy.signal import welch , decimate , detrend
from dataclasses import dataclass
import numpy as np
import sys
EPS=sys.float_info.epsilon

def compute_PSD(signals:np.ndarray,fs:int=250,q:int=2,nperseg:int=250*30,noverlap:int=250*20):
    """ Compute the power spectral density of the signal with Welch's method.
        with a decimate factor of q. if q=1, no decimation is performed.
    Args:
        signals (no): _description_
        fs (int, optional): _description_. Defaults to 250.
        q (int, optional): _description_. Defaults to 2.
        nperseg (_type_, optional): _description_. Defaults to 250*30.
        noverlap (_type_, optional): _description_. Defaults to 250*20.

    Returns:
        _type_: _description_
    """
    if q > 1:
        signals = decimate(signals, q, axis=1)
    signals = detrend(signals,type='constant')

    signals = signals - np.mean(signals, axis=1, keepdims=True)
    f,Sxxs= welch(signals,fs=int(fs/q),nperseg=nperseg,noverlap=noverlap)
    return f,Sxxs


def freq_to_mel(freq):
    return 250 * np.log10(1 + freq / 3)
def mel_to_freq(mel):
    return 3 * (10**(mel / 250) - 1)



def get_filter_points(fmin, fmax, n_mels,fft_size, fs=250):
    """ Compute the filter points ."""
    # Convert Hz to Mel
    min_mel = freq_to_mel(fmin)
    max_mel = freq_to_mel(fmax)
    # Equally spaced in Mel scale
    mels = np.linspace(min_mel, max_mel, n_mels)
    # Convert Mel to Hz
    freqs = mel_to_freq(mels)
    # Convert Hz to fft bin number
    return np.floor((fft_size + 1) * freqs / fs).astype(int) , freqs

def get_filters(filter_points, FFT_size):
    """ Compute the mel filters."""
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def energy_normalization(filters:np.array,mel_freqs:np.ndarray,n_mels:int):
    """ Energy normalization of the mel filters."""
    enorm = 2.0 / (mel_freqs[2:n_mels] - mel_freqs[:n_mels-2])
    filters *= enorm[:, np.newaxis]
    return filters
@dataclass
class Melfilterbank:
    """ compute the mel filterbank and apply it to the PSD"""
    n_mels: int = 40 
    fmin: float = 0.0
    fmax: float = 125.0
    n_fft: int = 250*30
    fs: int = 250
    def __post_init__(self):
        filter_points, mel_freqs=get_filter_points(self.fmin,self.fmax,self.n_mels,self.n_fft)
        self.filters = get_filters(filter_points, self.n_fft)
        self.filters = energy_normalization(self.filters,mel_freqs,self.n_mels)

    def __call__(self, PSD:np.ndarray):
        """ apply the mel filterbank to the PSD

        Args:
            PSD (np.ndarray): 

        Returns:
            _type_: mel spectrogram
        """
        return 10.0* np.log10(np.dot(self.filters, np.transpose(PSD))+EPS)

if __name__ =='__main__':
    from AD_vibration.data_loader.data_loader import DataLoader, Sensor, get_processed_PSD
    from datetime import datetime, timedelta
    import numpy as np
    import matplotlib.pyplot as plt

    Sxxs,dts=get_processed_PSD()
    mel = Melfilterbank(n_mels=400, fmin=0.0, fmax=125.0, n_fft=250*30, fs=125)
    plt.figure(figsize=(15,4))
    for n in range(mel.filters.shape[0]):
        plt.plot(np.linspace(0,62.5,3751),mel.filters[n])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter weight')
    plt.show()
    plt.close()
    mel_fb = mel(Sxxs[0])
    fig,ax= plt.subplots(nrows=2,figsize=(15,4))
    ax[1].plot(mel_fb)
    ax[0].plot(np.log(Sxxs[0].T+EPS))
    plt.show()
    plt.close()





