import numpy as np
import os
import matplotlib.pyplot as plt

import sys
my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa: E402
from get_fpath import get_fpath  # noqa: E402
from ProcessBar import ProcessBar  # noqa: E402


def file_reader(data_dir):
    """Read spectrum files under given directory
    Args: data_dir:
        Returns: sample generator, [samples,label_onehots] """
    if isinstance(data_dir, list):
        dir_list = data_dir
    else:
        dir_list = [data_dir]

    fpath_all = []
    for sub_set_dir in dir_list:
        fpath_all_tmp = get_fpath(sub_set_dir, '.npy', is_absolute=True)
        fpath_all.extend(fpath_all_tmp)

    pb = ProcessBar(len(fpath_all))
    for fpath in fpath_all:
        pb.update()
        fea = np.load(fpath)
        yield fea


def cal_norm_params(fea_dir):
    """
    """
    mean_37d = 0
    n_sample = 0
    for samples in file_reader(fea_dir):
        mean_37d = mean_37d + np.sum(samples, axis=0)
        n_sample = n_sample + samples.shape[0]
    mean_37d = mean_37d/n_sample
    mean_1d = np.mean(mean_37d)

    std_37d = 0
    std_1d = 0
    for samples in file_reader(fea_dir):
        std_37d = std_37d + np.sum((samples-mean_37d)**2, axis=0)
        std_1d = std_1d + np.sum((samples-mean_1d)**2)
    std_37d = np.sqrt(std_37d/n_sample)
    std_1d = np.sqrt(std_1d/n_sample)

    return [[mean_1d, std_1d], [mean_37d, std_37d]]


if __name__ == '__main__':

    # calculate the normalization params of mct training dataset
    room_all = ['Anechoic', 'A', 'B', 'C', 'D']
    reverb_room_all = ['A', 'B', 'C', 'D']
    data_dir = '../../Data/v1/GCC_PHAT_noisy/train'
    norm_param_dir = 'norm_params'

    # act
    dataset_room_dir_all = [os.path.join(data_dir, room) for room in room_all]
    print('act\n')
    [print(item) for item in dataset_room_dir_all]

    [[mean_1d, std_1d],
     [mean_37d, std_37d]] = cal_norm_params(dataset_room_dir_all)

    np.save(os.path.join(norm_param_dir, 'act_37d.npy'), [mean_37d, std_37d])

    np.save(os.path.join(norm_param_dir, 'act_1d.npy'), [mean_1d, std_1d])

    # mct
    fig, ax = plt.subplots(1, 1)
    tau = np.arange(-18, 19)
    bar_width = 0.2

    for room_i, room_tar in enumerate(reverb_room_all):
        room_mct_all = [room for room in room_all if room != room_tar]
        dataset_room_dir_all = [os.path.join(data_dir, room)
                                for room in room_mct_all]
        print(f'mct {room_tar}\n')
        [print(item) for item in dataset_room_dir_all]
        print('\n\n')

        [[mean_1d, std_1d],
         [mean_37d, std_37d]] = cal_norm_params(dataset_room_dir_all)

        np.save(os.path.join(norm_param_dir, f'mct_{room_tar}_37d.npy'),
                [mean_37d, std_37d])

        np.save(os.path.join(norm_param_dir, f'mct_{room_tar}_1d.npy'),
                [mean_1d, std_1d])

        ax.errorbar(x=tau+(room_i-1)*bar_width, y=mean_37d, yerr=std_37d,
                    label=room_tar)
    ax.errorbar(x=tau, y=mean_37d, yerr=std_37d)
    ax.set_xlabel('Delay(sample)')
    plot_tools.savefig(fig, fig_name='norm_params_mct.png')
