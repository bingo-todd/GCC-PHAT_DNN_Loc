import numpy as np
import os
import sys

sys.path.append('../../../Models/GCC_PHAT_DNN/')
from LocDNN import LocDNN  # noqa: E402
from utils import file_reader  # noqa: E402

data_dir_base = '../../WaveLoc/Data'
reverb_room_all = ['A', 'B', 'C', 'D']
n_reverb_room = 4
chunk_size = 25
n_test = 3
gpu_index = 1


def evaluate_mct(model_dir_base, norm_params_dir):

    rmse_all = np.zeros((n_test, n_reverb_room))

    for room_i, room in enumerate(reverb_room_all):

        norm_params_fpath = os.path.join(norm_params_dir,
                                         f'mct_{room}_1d.npy')
        reader_args = {'norm_params_fpath': norm_params_fpath,
                       'batch_size': 128}

        model_dir = os.path.join(model_dir_base, room)
        model_config_fpath = os.path.join(model_dir, 'config.cfg')
        model = LocDNN(file_reader, reader_args,
                       config_fpath=model_config_fpath, gpu_index=gpu_index)
        model.load_model(model_dir)

        for test_i in range(n_test):
            dataset_dir_test = os.path.join(
                data_dir_base,
                f'v{test_i+1}/GCC_PHAT/test/{room}')
            rmse_all[test_i, room_i] = model.evaluate_chunk(
                                            dataset_dir_test,
                                            chunk_size=chunk_size)

    return rmse_all


def evaluate_act(model_dir, norm_params_fpath):
    rmse_all = np.zeros((n_test, n_reverb_room))

    reader_args = {'norm_params_fpath': norm_params_fpath,
                   'batch_size': 128}

    model_config_fpath = os.path.join(model_dir, 'config.cfg')
    model = LocDNN(file_reader, reader_args,
                   config_fpath=model_config_fpath, gpu_index=gpu_index)
    model.load_model(model_dir)

    for room_i, room in enumerate(reverb_room_all):
        for test_i in range(n_test):
            dataset_dir_test = os.path.join(
                                data_dir_base,
                                f'v{test_i+1}/GCC_PHAT/test/{room}')
            rmse_all[test_i, room_i] = model.evaluate_chunk(
                                            dataset_dir_test,
                                            chunk_size=chunk_size)
    return rmse_all


if __name__ == '__main__':
    train_strategy = sys.argv[1]
    model_dir = sys.argv[2]
    norm_params_fpath = sys.argv[3]
    if train_strategy == 'mct':
        rmse_all = evaluate_mct(model_dir, norm_params_fpath)
    elif train_strategy == 'act':
        rmse_all = evaluate_act(model_dir, norm_params_fpath)
    else:
        raise Exception()

    if len(sys.argv) >= 5:
        result_fpath = sys.argv[4]
        np.save(result_fpath, rmse_all)

    print(train_strategy)
    print(rmse_all)
    print('mean:', np.mean(rmse_all, axis=0))
    print('std:', np.std(rmse_all, axis=0))
