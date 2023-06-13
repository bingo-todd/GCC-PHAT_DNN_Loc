import matplotlib.pyplot as plt
import numpy as np
import os
from BasicTools import plot_tools, get_file_path


data_dir_base = '../../Data'
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
max_delay = 18


def file_reader(data_dir):
    """Read spectrum files under given directory
    Args:
        data_dir:
    Returns:
        sample generator, [samples,label_onehots]
    """
    if isinstance(data_dir, list):
        dir_list = data_dir
    else:
        dir_list = [data_dir]

    for sub_set_dir in dir_list:
        fea_fpath_list = get_file_path(sub_set_dir, '.npy')
        for fea_fpath in fea_fpath_list:
            fea_fpath_abs = os.path.join(sub_set_dir, fea_fpath)
            fea = np.load(fea_fpath_abs)
            yield fea


def cal_norm_coef(fea_dir):
    """
    """
    mean_37d = 0
    n_sample = 0
    for samples in file_reader(fea_dir):
        mean_37d = mean_37d + np.sum(samples, axis=0)
        n_sample = n_sample+samples.shape[0]
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


if __name__ == "__main__":
    # calculate the normalization params of mct training dataset
    dataset_dir = os.path.join(data_dir_base, 'v1/GCC_PHAT/train/reverb')
    norm_param_dir = '../norm_coef'

    fig, ax = plt.subplots(1, 1)
    tau = np.arange(-18, 19)
    bar_width = 0.2
    for room_i, room_tar in enumerate(room_all[1:]):
        mct_room_all = [room for room in room_all if room != room_tar]

        dataset_room_dir_all = [os.path.join(dataset_dir, room)
                                for room in mct_room_all]
        [[mean_1d, std_1d],
         [mean_37d, std_37d]] = cal_norm_coef(dataset_room_dir_all)

        param_name = 'mct_{}'.format(room_tar)
        # np.save(os.path.join(norm_param_dir, '{}_37d.npy'.format(param_name)),
        #         [mean_37d, std_37d])

        # np.save(os.path.join(norm_param_dir, '{}_1d.npy'.format(param_name)),
        #         [mean_1d, std_1d])

        ax.errorbar(x=tau+(room_i-1)*bar_width, y=mean_37d, yerr=std_37d,
                    label=room_tar)
    ax.set_xlabel('Delay(sample)')
    ax.legend()
    plot_tools.savefig(fig, fig_name='norm_coef.png', fig_dir='../images')

    # normalization parameters of all train data
    if True:
        dataset_room_dir_all = [os.path.join(dataset_dir, room)
                                for room in room_all]
        [[mean_1d, std_1d],
        [mean_37d, std_37d]] = cal_norm_coef(dataset_room_dir_all)

        np.save(os.path.join(norm_param_dir, 'all_room_37d.npy'),
                [mean_37d, std_37d])

        np.save(os.path.join(norm_param_dir, 'all_room_1d.npy'),
                [mean_1d, std_1d])

        tau = np.arange(-18, 19)
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(x=tau, y=mean_37d, yerr=std_37d)
        ax.set_xlabel('Delay(sample)')
        plot_tools.savefig(fig, fig_name='norm_coef.png')
