import os
import matplotlib.pyplot as plt
import numpy as np
from BasicTools import plot_tools
from BasicTools.get_file_path import get_file_path


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


if __name__ == '__main__':
    """
    """
    data_dir = '../../../WaveLoc/Data'

    room = 'C'
    dataset_dir = os.path.join(data_dir, f'v1/GCC_PHAT/train/{room}')

    norm_params_37d_fpath = f'../norm_params/mct_{room}_37d.npy'
    mean_37d, std_37d = np.load(norm_params_37d_fpath)

    norm_params_1d_fpath = f'../norm_params/mct_{room}_1d.npy'
    mean_1d, std_1d = np.load(norm_params_1d_fpath)

    sample_generator = file_reader(dataset_dir)
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)

    tau_axis = np.arange(-18, 19)
    for samples in sample_generator:
        samples_sep_norm = ((samples-mean_37d)/std_37d)
        ax1.plot(tau_axis, samples_sep_norm.T)
        ax1.set_title('separate_norm')
        ax1.set_xlabel('delay(sample)')
        # axes[0].set_ylim([-3,5])

        samples_overall_norm = (samples-mean_1d)/std_1d
        ax2.plot(tau_axis, samples_overall_norm.T)
        ax2.set_title('overal_norm')
        ax2.set_xlabel('delay(sample)')
        # axes[1].set_ylim([-3,5])

        plot_tools.savefig(fig1, fig_name='separate_norm_example.png',
                           fig_dir='../images/dataset')
        plot_tools.savefig(fig2, fig_name='overall_norm_example.png',
                           fig_dir='../images/dataset')
        break
