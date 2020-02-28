import numpy as np
import os
import sys
from LocDNN import LocDNN  
from utils import file_reader

data_dir_base = '../Data'
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']
n_reverb_room = 4
chunk_size = 25
n_test = 3
gpu_index = 1


def evaluate_mct(model_dir):
    rmse_all = np.zeros((n_test, n_reverb_room))
    for room_i, room in enumerate(reverb_room_all):
        model_dir_room = os.path.join(model_dir, room)
        config_fpath = os.path.join(model_dir_room, 'config.cfg')
        model = LocDNN(file_reader.file_reader, config_fpath, gpu_index)
        model.load_model(model_dir_room)

        for test_i in range(n_test):
            dataset_dir_test = os.path.join(
                data_dir_base,
                f'v{test_i+1}/GCC_PHAT/test/reverb/{room}')
            rmse_all[test_i, room_i] = model.evaluate_chunk(
                                            dataset_dir_test,
                                            chunk_size=chunk_size)
    return rmse_all


if __name__ == '__main__':
    model_dir = sys.argv[1]
    rmse_all = evaluate_mct(model_dir)

    with open(os.path.join(model_dir, 'result.txt'), 'w') as result_file:
        result_file.write(f'{rmse_all}')
        result_file.write('mean: {}\n'.format(np.mean(rmse_all, axis=0)))
        result_file.write('std: {}\n'.format(np.std(rmse_all, axis=0)))

    print('{}'.format(rmse_all))
    print('mean: {}'.format(np.mean(rmse_all, axis=0)))
    print('std: {}'.format(np.std(rmse_all, axis=0)))
