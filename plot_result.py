import matplotlib.pyplot as plt
import numpy as np
import os
import sys
my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa: E402

reverb_room_all = ['A', 'B', 'C', 'D']


def plot_train_process(record_fpath, ax=None, label=None):

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=[6, 4],
                               sharex=True, tight_layout=True)
    else:
        fig = None

    record_info = np.load(record_fpath)

    #
    cost_record_valid = record_info['cost_record_valid']
    rmse_record_valid = record_info['azi_rmse_record_valid']
    n_epoch = np.nonzero(cost_record_valid)[0][-1] + 1

    plot_settings = {'label': label, 'linewidth': 2}
    ax[0].plot(cost_record_valid[:n_epoch], **plot_settings)
    ax[1].plot(rmse_record_valid[:n_epoch], **plot_settings)

    ax[1].legend()

    ax[0].set_ylabel('Cross entrophy')
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('Epoch(n)')
    ax[1].set_xlabel('Epoch(n)')

    return fig


def plot_mct_train_process(model_dir):

    reverb_room_all = ['A', 'B', 'C', 'D']
    fig, ax = plt.subplots(1, 2, figsize=[6, 4], sharex=True,
                           tight_layout=True)

    for room_i, room in enumerate(reverb_room_all):
        record_fpath = os.path.join(model_dir, room, 'train_record.npz')
        plot_train_process(record_fpath, ax, label=room)
    return fig


def plot_evaluate_result(result_fpath_all, label_all):

    mean_std_all = []
    for result_fpath in result_fpath_all:
        rmse_multi_test = np.load(result_fpath)
        rmse_mean = np.mean(rmse_multi_test, axis=0)
        rmse_std = np.std(rmse_multi_test, axis=0)
        mean_std_all.append([rmse_mean, rmse_std])

    fig = plot_tools.plot_bar(*mean_std_all, legend=label_all,
                              xticklabels=reverb_room_all,
                              xlabel='Room', ylabel='RMSE($^o$)',
                              ylim=[0, 4])
    return fig


def plot_evaluation(result_fpath_all, legend_all):
    mean_std_all = []
    for result_fpath in result_fpath_all:
        rmse_multi_test = np.load(result_fpath)

        rmse_mean = np.mean(rmse_multi_test, axis=0)
        rmse_std = np.std(rmse_multi_test, axis=0)
        mean_std_all.append([rmse_mean, rmse_std])

    fig = plot_tools.plot_bar(*mean_std_all, legend=legend_all,
                              xticklabels=reverb_room_all,
                              xlabel='Room', ylabel='RMSE($^o$)',
                              ylim=[0, 4])
    return fig


if __name__ == '__main__':

    if True:
        mct_model_dir = 'models/mct_37dnorm'
        fig = plot_mct_train_process(mct_model_dir)
        plot_tools.savefig(fig, fig_name='train_process_mct_37dnorm.png',
                           fig_dir='images/training')

        mct_model_dir = 'models/mct_1dnorm'
        fig = plot_mct_train_process(mct_model_dir)
        plot_tools.savefig(fig, fig_name='train_process_mct_1dnorm.png',
                           fig_dir='images/training')
