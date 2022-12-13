import matplotlib.pyplot as plt
import numpy as np

def plot_example_psd(psd: np.ndarray,fs=None,ax=None,**kwargs):
    """Plot an example of a PSD

    Args:
        psd (np.ndarray): PSD of a single example

    Returns:
        _type_: plot
    """
    if ax is None:
        ax = plt.gca()
    if fs is None:
        freq = np.linspace(0,len(psd),len(psd))
    else:
        freq = np.linspace(0,fs,len(psd))
    ax.plot(freq,psd,**kwargs)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    return ax