from multiprocessing import Process
import os
import sys
import configparser
from LocDNN import LocDNN
from utils import file_reader

room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']

data_dir = os.path.expanduser('../Data')
train_set_dir = os.path.join(data_dir, 'v1/GCC_PHAT/train/reverb')
valid_set_dir = os.path.join(data_dir, 'v1/GCC_PHAT/valid/reverb')

model_basic_settings = {'fea_len': 37,
                        'is_norm': 'True',
                        'is_dropout': 'True',
                        'n_azi': 37}
gpu_index = 1


def train_mct(room_tar, model_dir, norm_coef_fpath):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    room_mct_all = [room for room in room_all if room != room_tar]

    config = configparser.ConfigParser()
    config['model'] = {'model_dir': model_dir,
                       'norm_coef_fpath': norm_coef_fpath,
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
        config.write(config_file)

    model = LocDNN(file_reader.file_reader, config_fpath, gpu_index)
    model.train_model(model_dir)


if __name__ == '__main__':
	thread_all = []
	for room_tar in reverb_room_all:
		model_dir = f'models_{test_i}/mct_37dnorm'
		norm_coef_fpath = 'norm_coef/mct_Room_D_37d.npy'
		thread = Process(target=train_mct, args=(room_tar, model_dir, norm_coef_fpath))
		thread.start()
		thread_all.append(thread)

	[thread.join() for thread in thread_all]
