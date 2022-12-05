from scipy.signal import decimate
from scipy.signal import welch
from dataclasses import dataclass
import numpy as np

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
    signals = signals - np.mean(signals, axis=1, keepdims=True)
    f,Sxxs= welch(signals,fs=int(fs/q),nperseg=nperseg,noverlap=noverlap)
    return f,Sxxs


def freq_to_mel(freq):
    return 250 * np.log10(1 + freq / 3)
def mel_to_freq(mel):
    return 3 * (10**(mel / 250) - 1)

def mel_filterbank(n_mels:int=40, fmin:float=0.0, fmax:float=None, n_fft:int=2048, fs:int=250): 
    if fmax is None:
        fmax = fs / 2
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = mel_to_freq(mels)
    return np.floor((n_fft + 1) * freqs / fs) , freqs

def get_filter_points(fmin, fmax, n_mels,fft_size, fs=250):
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
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def energy_normalization(filters:np.array,mel_freqs:np.ndarray,n_mels:int):
    enorm = 2.0 / (mel_freqs[2:n_mels] - mel_freqs[:n_mels-2])
    filters *= enorm[:, np.newaxis]
    return filters
@dataclass
class mel_filterbank:
    n_mels: int = 40 
    fmin: float = 0.0
    fmax: float = 125.0
    n_fft: int = 250*30
    fs: int = 250
    def __post_init__(self):
        mel_filterbank(self.n_mels,self.fmin,self.fmax,self.n_fft,self.fs)
        filter_points, mel_freqs=get_filter_points(self.fmin,self.fmax,self.n_mels,self.n_fft)
        self.filters = get_filters(filter_points, self.n_fft)
        self.filters = energy_normalization(self.filters,mel_freqs,self.n_mels)

    def __call__(self, PSD):
        return 10.0* np.log10(np.dot(self.filters, PSD))





