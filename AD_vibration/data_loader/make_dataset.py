from data_loader import DataLoader, Sensor
from datetime import datetime, timedelta
import numpy as np
from AD_vibration.utils import get_config
from pathlib import Path
from AD_vibration.features.build_features import compute_PSD

def main():
    frame= timedelta(minutes= 20)
    step = timedelta(minutes= 15)
    sensor = Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    data_loader = DataLoader(sensor=sensor)
    start = datetime(2022, 3,29,15,0,0) 
    end = datetime(2022, 6,25,15,1,0)

    path_psd = Path(get_config()['PATH']['data_root_processed_PSD'])/'PSDs.npy'
    path_dts = Path(get_config()['PATH']['data_root_processed_PSD'])/'dts.npy'
    ### parameters defined
    PSDs,dts=[],[]
    dt= start
    while dt<end:
        try:
            data = data_loader._load_single(start=dt,end = dt+frame)
            if data is None:
                dt+=step
                continue

            dts.append(dt)
            PSDs.append(compute_PSD(data,fs=250,q=2,window='hann',nperseg=30*250,noverlap=20*250))
            dt+=step
        except:
            dt+=step
            continue

    np.save(path_psd,PSDs)
    np.save(path_dts,dts)

if __name__ == '__main__':
    main()
