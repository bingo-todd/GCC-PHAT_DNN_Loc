import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import numpy as np
from BasicTools import plot_tools
plt.rcParams.update({"font.size": "12"})
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']
n_test = 3

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2, figsize=[6, 3], tight_layout=True)
    plot_settings = {'linewidth': 2}
    n_epoch_max = 0

    model_dir_base = sys.argv[1]  # '../models/mct_37dnorm'
    for room in reverb_room_all:
        # learning curve
        record_fpath = f'{model_dir_base}/{room}/train_record.npz'
        record_info = np.load(record_fpath)
        cost_record_valid = record_info['cost_record_valid']
        n_epoch = np.nonzero(cost_record_valid)[0][-1] + 1
        if n_epoch > n_epoch_max:
            n_epoch_max = n_epoch
        ax[0].plot(cost_record_valid[:n_epoch], **plot_settings,
                   label=room[-1])

    # rmse
    rmse_all = np.load(f'{model_dir_base}/result.npy')
    rmse_mean_all = np.mean(rmse_all, axis=0)
    rmse_std_all = np.std(rmse_all, axis=0)

    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].legend()
    ax[0].set_ylabel('Cross entrophy')
    ax[0].set_xlabel('Epoch(n)')
    ax[0].set_title('Learning Curve')

    ax[1].bar([room[-1] for room in reverb_room_all], rmse_mean_all,
              yerr=rmse_std_all/np.sqrt(n_test))
    plt.xticks(rotation=45)
    ax[1].set_xlabel('Room')
    ax[1].set_ylabel('RMSE')
    ax[1].set_title('Test')
    man_std = [[rmse_mean_all[i], rmse_std_all[i]] for i in range(4)]
    plot_tools.savefig(fig, 'result.png', model_dir_base)
