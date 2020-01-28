import numpy as np
import copy
import os
import sys
my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
from get_fpath import get_fpath  # noqa 402
import plot_tools  # noqa: E402


reverb_room_all = ['A', 'B', 'C', 'D']


def file_reader(dataset_dir, norm_params_fpath, batch_size=-1,
                is_shuffle=True):
    """Read spectrum files under given directory
    Args:
        dataset_dir:
    Returns:
        sample generator, [samples,label_onehots]
    """
    n_azi = 37
    fea_len = 37

    if isinstance(dataset_dir, list):
        dir_all = dataset_dir
    else:
        dir_all = [dataset_dir]

    mean, std = np.load(norm_params_fpath)

    fpath_all = []
    for dir_tmp in dir_all:
        fpath_all_tmp = get_fpath(dir_tmp, suffix='.npy', is_absolute=True)
        fpath_all.extend(fpath_all_tmp)

    if is_shuffle:
        np.random.shuffle(fpath_all)  # randomize files order

    if len(fpath_all) < 1:
        raise Exception('folder is empty: {}'.format(dataset_dir))

    if batch_size > 0:
        x_all = np.zeros((0, fea_len))
        y_all = np.zeros((0, n_azi))

    for fpath in fpath_all:
        fea_file_all = np.load(fpath)
        x_file_all = np.divide(fea_file_all-mean, std)
        n_sample_tmp = x_file_all.shape[0]

        fname, _ = os.path.basename(fpath).split('.')
        azi, _ = map(int, fname.split('_'))
        y_file_all = np.zeros((n_sample_tmp, n_azi))
        y_file_all[:, azi] = 1

        if batch_size > 0:
            x_all = np.concatenate((x_all, x_file_all), axis=0)
            y_all = np.concatenate((y_all, y_file_all), axis=0)
            while(x_all.shape[0] > batch_size):
                x_batch = copy.deepcopy(x_all[:batch_size])
                y_batch = copy.deepcopy(y_all[:batch_size])

                x_all = x_all[batch_size:]
                y_all = y_all[batch_size:]

                yield [x_batch, y_batch]
        else:
            yield [x_file_all, y_file_all]
