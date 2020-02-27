"""
synthesize spatial recordings
"""

import numpy as np
import os
from multiprocessing import Process
from BasicTools import ProcessBarMulti
from BasicTools import wav_tools
from BasicTools.Filter_GPU import Filter_GPU
from BasicTools.get_fpath import get_fpath


TIMIT_dir = os.path.expanduser('~/Work_Space/Data/TIMIT_wav')
brirs_dir = 'brirs_aligned'
data_dir = '../Data'

room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
n_room = 5
n_azi = 37  # -90~90 in step of 5
n_wav_per_azi_all = {'train': 24,
                     'valid': 6,
                     'test': 15}

fs = 16e3
frame_len = int(20e-3*fs)
shift_len = int(10e-3*fs)


def load_brirs(room):
    """load brirs of given room
    Args:
        room: room name from ['Anechoic','A','B','C','D']
    """
    brirs_fpath = os.path.join(brirs_dir, f'{room}.npy')
    brirs = np.load(brirs_fpath)
    return brirs


def truncate_silence(x):
    """ trip off slient frames in begining and end(only one frame)
    """
    vad_flag = wav_tools.vad(x, frame_len, shift_len)
    if not vad_flag[0]:  # silence in the first frame
        x = x[frame_len:]
    if not vad_flag[-1]:  # silence in the last frame
        x = x[:-frame_len]
    return x


def syn_record(src_fpath_all, set_dir, n_wav_per_azi, task_i, pb):
    """synthesize spatial recordings as well corresponding direct sound for
    each set
    """
    filter_gpu = Filter_GPU(gpu_index=1)

    brirs_direct = load_brirs('Anechoic')
    wav_count = 0
    for room in room_all:
        direct_dir = os.path.join(set_dir, 'direct', room)
        os.makedirs(direct_dir, exist_ok=True)
        rever_dir = os.path.join(set_dir, 'reverb', room)
        os.makedirs(rever_dir, exist_ok=True)

        brirs_room = load_brirs(room)
        for azi_i in range(n_azi):
            for i in range(n_wav_per_azi):
                pb.update(task_i)
                src_fpath = src_fpath_all[wav_count]
                wav_count = wav_count+1

                src, fs = wav_tools.read_wav(src_fpath)
                src = truncate_silence(src)

                direct = filter_gpu.brir_filter(src, brirs_direct[azi_i])
                # direct = wav_tools.brir_filter(src, brirs_direct[azi_i])
                direct_fpath = os.path.join(direct_dir, f'{azi_i}_{i}.wav')
                wav_tools.write_wav(direct, fs, direct_fpath)

                reverb = filter_gpu.brir_filter(src, brirs_room[azi_i])
                # reverb = wav_tools.brir_filter(src, brirs_room[azi_i])
                reverb_fpath = os.path.join(rever_dir, f'{azi_i}_{i}.wav')
                wav_tools.write_wav(reverb, fs, reverb_fpath)


def gen_dataset(dir_path, set_type_all):

    n_wav_train = n_azi * n_room * n_wav_per_azi_all['train']
    n_wav_valid = n_azi * n_room * n_wav_per_azi_all['valid']
    n_wav_test = n_azi * n_room * n_wav_per_azi_all['test']
    # prepare sound source
    # train and validate
    if not os.path.exists('fpath_TIMIT_train_all.npy'):
        TIMIT_train_dir = os.path.join(TIMIT_dir, 'TIMIT/TRAIN')
        src_fpath_all = get_fpath(TIMIT_train_dir, '.wav', 
                                  is_absolute=True)
        np.save('fpath_TIMIT_train_all.npy', src_fpath_all)
    src_fpath_all = np.load('fpath_TIMIT_train_all.npy')
    print('train', len(src_fpath_all))
    print('train+valid', n_wav_train+n_wav_valid)
    np.random.shuffle(src_fpath_all)
    src_fpath_train_all = src_fpath_all[:n_wav_train]
    src_fpath_valid_all = src_fpath_all[n_wav_train:]

    # test
    if not os.path.exists('fpath_TIMIT_test_all.npy'):
        TIMIT_test_dir = os.path.join(TIMIT_dir, 'TIMIT/TEST')
        src_fpath_test_all = get_fpath(TIMIT_test_dir, '.wav', 
                                  is_absolute=True)
        np.save('fpath_TIMIT_test_all.npy', src_fpath_test_all)
    src_fpath_test_all = np.load('fpath_TIMIT_test_all.npy')
    print('test', len(src_fpath_test_all))
    print('test', n_wav_test)
    np.random.shuffle(src_fpath_test_all)

    # np.save(os.path.join(dir_path, 'src_fpath_all.npy'),
    #         [src_fpath_train_all, src_fpath_valid_all, src_fpath_test_all])

    src_fpath_all = (src_fpath_train_all,
                     src_fpath_valid_all,
                     src_fpath_test_all)

    n_wav_all = [len(src_fpath_all[i]) for i in range(len(set_type_all))]
    pb = ProcessBarMulti(n_wav_all, desc_all=set_type_all)
    proc_all = []
    for i, set_type in enumerate(set_type_all):
        print(set_type)
        set_dir = os.path.join(dir_path, set_type)
        proc = Process(target=syn_record,
                       args=(src_fpath_all[i], set_dir,
                             n_wav_per_azi_all[set_type],
                             str(i), pb))
        proc.start()
        proc_all.append(proc)
    [proc.join() for proc in proc_all]


if __name__ == '__main__':

    # train dataset and validation dataset
    dataset_dir = os.path.join(data_dir, 'v1')
    os.makedirs(dataset_dir, exist_ok=True)
    gen_dataset(dir_path=dataset_dir,
                set_type_all=['train', 'valid'])

    # test dataset
    for i in range(1, 5):
        dataset_dir = os.path.join(data_dir, f'v{i}')
        os.makedirs(dataset_dir, exist_ok=True)
        gen_dataset(dir_path=dataset_dir, set_type_all=['test'])
