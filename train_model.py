import os
import sys
import configparser
from LocDNN import LocDNN
from utils import file_reader


room_all = ['Anechoic', 'A', 'B', 'C', 'D']
reverb_room_all = ['A', 'B', 'C', 'D']

data_dir = os.path.expanduser('../../WaveLoc/Data')
train_set_dir = os.path.join(data_dir, 'v1/GCC_PHAT/train')
valid_set_dir = os.path.join(data_dir, 'v1/GCC_PHAT/valid')

model_basic_settings = {'fea_len': 37,
                        'is_norm': 'True',
                        'is_dropout': 'True',
                        'n_azi': 37}

gpu_index = 1


def train_mct(room_tar, model_dir, norm_params_fpath):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    room_mct_all = [room for room in room_all if room != room_tar]

    config = configparser.ConfigParser()
    config['model'] = {'model_dir': model_dir,
                       'norm_params_fpath': norm_params_fpath,
                       **model_basic_settings}

    config['train'] = {'batch_size': '128',
                       'max_epoch': '50',
                       'is_print_log': 'True',
                       'train_set_dir': ';'.join(
                                        [os.path.join(train_set_dir, room)
                                         for room in room_mct_all]),
                       'valid_set_dir': ';'.join(
                                        [os.path.join(valid_set_dir, room)
                                         for room in room_mct_all])}

    config_fpath = os.path.join(model_dir, 'config.cfg')
    with open(config_fpath, 'w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    reader_args = {'norm_params_fpath': norm_params_fpath,
                   'batch_size': 128}

    model = LocDNN(file_reader, reader_args, config_fpath, gpu_index)
    model.train_model(model_dir)


def train_act(model_dir, norm_params_fpath):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config = configparser.ConfigParser()
    config['model'] = {'model_dir': model_dir,
                       'norm_params_fpath': norm_params_fpath,
                       **model_basic_settings}

    config['train'] = {'batch_size': '128',
                       'max_epoch': '50',
                       'is_print_log': 'True',
                       'train_set_dir': ';'.join(
                                        [os.path.join(train_set_dir, room)
                                         for room in room_all]),
                       'valid_set_dir': ';'.join(
                                        [os.path.join(valid_set_dir, room)
                                         for room in room_all])}

    config_fpath = os.path.join(model_dir, 'config.cfg')
    with open(config_fpath, 'w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    reader_args = {'norm_params_fpath': norm_params_fpath,
                   'batch_size': 128}

    model = LocDNN(file_reader, reader_args, config_fpath, gpu_index)
    model.train_model(model_dir)


if __name__ == '__main__':
    train_strategy = sys.argv[1]
    if train_strategy == 'mct':
        room_tar = sys.argv[2]
        model_dir = sys.argv[3]
        norm_params_fpath = sys.argv[4]
        train_mct(room_tar, model_dir, norm_params_fpath)
    elif train_strategy == 'act':
        model_dir = sys.argv[2]
        norm_params_fpath = sys.argv[3]
        train_act(model_dir, norm_params_fpath)
