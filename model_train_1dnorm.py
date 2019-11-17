import os
import matplotlib.pyplot as plt
import configparser
from LocDNN import LocDNN
import multiprocessing

def main_sub(i):
	# basic variables
	room_list = ['Anechoic','A','B','C','D']
	train_set_dir_base = '/mnt/hd6t/songtao/Localize/WavLoc/Data/v1/GCC_PHAT/train/reverb'
	valid_set_dir_base = '/mnt/hd6t/songtao/Localize/WavLoc/Data/v1/GCC_PHAT/valid/reverb'

	model_dir = 'models/models_1dnorm'
	tar_room = room_list[1+i]

	model_dir = os.path.join(model_dir,'mct_not_{}'.format(tar_room))
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# filter out tar_room from room_list
	mct_filter = lambda room : room != tar_room

	config = configparser.ConfigParser()
	config['model']={'model_dir':model_dir,
	                 'frame_len':320,
					 'overlap_len':160,
	                 'is_norm':True,
					 'norm_params_fpath':'norm_params/mct_not_{}_1d.npy'.format(tar_room),
					 'is_dropout':True,
	                 'azi_num':37}
	config['train']={'batch_size':128,
	                 'max_epoch':50,
					 'is_print_log':True,
	                 'train_set_dir':';'.join([os.path.join(train_set_dir_base,room)
									 			 for room in filter(mct_filter,room_list)]) ,
	                 'valid_set_dir':';'.join([os.path.join(valid_set_dir_base,room)
												 for room in filter(mct_filter,room_list)])}

	config_fpath = os.path.join(model_dir,'mct_not_{}.cfg'.format(tar_room))
	with open(config_fpath,'w') as config_file:
	    if config_file is None:
	        raise Exception('fail to create file')
	    config.write(config_file)

	model = LocDNN(config_fpath,gpu_index=0)
	model.train_model()


if __name__ == '__main__':
	pool = multiprocessing.Pool(4)
	pool.map(main_sub,range(4))
