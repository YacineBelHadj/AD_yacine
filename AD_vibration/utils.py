import configparser
from pathlib import Path


def get_config(path_relative='config.ini'):
    cfg= configparser.ConfigParser()
    path_abs = Path(__file__).parent.parent / path_relative
    cfg.read(path_abs)
    return cfg

if __name__=='__main__':
    config = configparser.ConfigParser()
    path_abs = Path(__file__).parent.parent 
    print(path_abs)
    config['PATH'] = {'data_root':  Path('/media/yacine/Intenso/RawData').as_posix(),
                      'data_root_processed_PSD': (path_abs / Path('data/processed/PSD')).as_posix()}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
