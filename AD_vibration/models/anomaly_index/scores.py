
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class MahalanobisDistance(BaseEstimator, TransformerMixin):
    cov : np.ndarray = None
    mean : np.ndarray = None
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """Fit the Mahalanobis distance model
        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to fit the model
        """ 
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.cov = np.cov(data.T)
        self.mean = np.mean(data, axis=0)

    def transform(self, data: Union[np.ndarray, pd.DataFrame]):
        """Compute the Mahalanobis distance
        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Data to fit the model
        Returns
        -------
        distance : np.ndarray
            Mahalanobis distance
        """ 
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        distance = np.sqrt(np.diag((data - self.mean) @ np.linalg.inv(self.cov) @ (data - self.mean).T))
        return distance

def model_confidence(sensor:int,y_pred:np.ndarray):
    """ Compute the confidence of the model
    Parameters
    ----------
    sensor : int
        Sensor number
    y_pred : np.ndarray
        Predicted labels
    Returns
    -------
    confidence : np.ndarray
        Confidence of the model
    """
    confidence = y_pred[:,sensor]
    return confidence
    