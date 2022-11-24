from AD_structure.src.data.data_loader import utils 
import numpy as np
from pathlib import Path
from datetime import datetime,  timedelta

def test_append_dict():
    dict1 = {'a':np.array([1,2,3]),'b':np.array([4,5,6])}
    dict2 = {'a':np.array([7,8,9]),'b':np.array([10,11,12])}
    dict3 = {'a':np.array([1,2,3,7,8,9]),'b':np.array([4,5,6,10,11,12])}
    assert utils.append_dict(dict1,dict2) == dict3

