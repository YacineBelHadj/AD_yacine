import numpy as np

def dict_to_numpy(dict_signal:dict):
    """Convert dictionary to numpy array
    Parameters
    ----------
    dict_signal : dict
        dictionary to be converted to numpy array
    Returns
    -------
    signals : np.ndarray
        numpy array of signals
    channels : list
        list of channels
    """
    channels = list(dict_signal.keys())
    signals = np.array(list(dict_signal.values()))
    return signals, channels