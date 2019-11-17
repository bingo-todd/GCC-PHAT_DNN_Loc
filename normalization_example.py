import matplotlib.pyplot as plt
import numpy as np
import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir,'basic_tools/basic_tools'))
import gcc
import process_bar
import plot_tools
import wav_tools

from cal_fea import file_reader


if __name__ == '__main__':
    """
    """
    room = 'C'
    dataset_dir = ('/mnt/hd6t/songtao/Localize/WavLoc/Data/v1/'
                            'GCC_PHAT/train/reverb/{}'.format(room))

    norm_params_37d_fpath = 'norm_params/mct_not_{}_37d.npy'.format(room)
    mean_37d,std_37d = np.load(norm_params_37d_fpath)

    norm_params_1d_fpath = 'norm_params/mct_not_{}_1d.npy'.format(room)
    mean_1d,std_1d = np.load(norm_params_1d_fpath)

    sample_generator = file_reader(dataset_dir)
    fig1,ax1 = plt.subplots(1,1)
    fig2,ax2 = plt.subplots(1,1)

    tau_axis = np.arange(-18,19)
    for samples in sample_generator:
      samples_sep_norm = ((samples-mean_37d)/std_37d)
      ax1.plot(tau_axis,samples_sep_norm.T);
      ax1.set_title('separate_norm')
      ax1.set_xlabel('delay(sample)')
      # axes[0].set_ylim([-3,5])

      samples_overall_norm = (samples-mean_1d)/std_1d
      ax2.plot(tau_axis,samples_overall_norm.T);
      ax2.set_title('overal_norm')
      ax2.set_xlabel('delay(sample)')
      # axes[1].set_ylim([-3,5])

      plot_tools.savefig(fig1,name='separate_norm.png')
      plot_tools.savefig(fig2,name='overall_norm.png')
      break
