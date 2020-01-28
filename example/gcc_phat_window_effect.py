import matplotlib.pyplot as plt
import numpy as np

import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import wav_tools  # noqa: E402
import gcc  # noqa: E402
import plot_tools  # noqa: E402


if __name__ == '__main__':
    """ Calculate gcc-phat using different window function
    window functions：rect,hann,hamm
    """

    x, fs = wav_tools.wav_read('0.wav')

    win_funcs = {'rectangle': np.ones,
                 'hanning': np.hanning,
                 'hamming': np.hamming}

    fig, ax = plt.subplots(1, 3, figsize=[8, 3])
    frame_i = 30
    for i, func_name in enumerate(win_funcs.keys()):
        gcc_phat = gcc.cal_gcc_phat(x[:, 0], x[:, 1], frame_len=320,
                                    win_f=win_funcs[func_name])
        ax[i].plot(np.arange(-319, 320), gcc_phat[frame_i])
        ax[i].set_xlabel('Delay(sample)')
        ax[i].set_title(func_name)
    plt.tight_layout()
    plot_tools.savefig(fig, fig_name='gcc_phat_window_effect.png',
                       fig_dir='../images/example')

    fig, ax = plt.subplots(1, 3, figsize=[8, 3])
    for i, func_name in enumerate(win_funcs.keys()):
        gcc_phat = gcc.cal_gcc_phat(x[:, 0], x[:, 1], frame_len=320,
                                    win_f=win_funcs[func_name],
                                    max_delay=18)
        ax[i].plot(np.arange(-18, 19), gcc_phat[frame_i])
        ax[i].set_xlabel('Delay(sample)')
        ax[i].set_title(func_name)
    plt.tight_layout()
    plot_tools.savefig(fig, fig_name='gcc_phat_window_effect2.png',
                       fig_dir='../images/example')
