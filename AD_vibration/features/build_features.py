from scipy.signal import decimate
from scipy.signal import welch
import numpy as np

def compute_PSD(signals,fs=250,q=2,nperseg=250*30,noverlap=250*20):
    signals = decimate(signals, q, axis=1)
    signals = signals - np.mean(signals, axis=1, keepdims=True)
    f,Sxxs= welch(signals,fs=int(fs/q),nperseg=nperseg,noverlap=noverlap)
    return f,Sxxs


def freq_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700)
def mel_to_freq(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(n_mels=40, fmin=0, fmax=8000, n_fft=2048, sr=16000): 
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = mel_to_freq(mels)
    bins = np.floor((n_fft + 1) * freqs / sr)
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = int(bins[m - 1])
        f_m = int(bins[m])
        f_m_plus = int(bins[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank

def mel_filterbank(n_mels:int=40, fmin:float=0.0, fmax:float=None, n_fft:int=2048, fs:int=250): 
    if fmax is None:
        fmax = fs / 2
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = mel_to_freq(mels)
    return np.floor((n_fft + 1) * freqs / fs) , freqs
    