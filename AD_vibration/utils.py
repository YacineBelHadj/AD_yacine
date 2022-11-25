import configparser
from pathlib import Path


def get_config(path_relative='config.ini'):
    cfg= configparser.ConfigParser()
    path_abs = Path(__file__).parent.parent / path_relative

    cfg.read(path_abs)
    return cfg

if __name__=='__main__':
    config = configparser.ConfigParser()
    config['PATH'] = {'data_root': '/media/yacine/Intenso/RawData'}
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
