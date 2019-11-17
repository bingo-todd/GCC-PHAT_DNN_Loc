import numpy as np
import matplotlib.pyplot as plt

import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir,'basic_tools/basic_tools'))
import plot_tools

rmse_1d = np.load('models/models_1dnorm/multi_test_result_chunk.npy')
rmse_1d_mean = np.mean(rmse_1d,axis=0)
rmse_1d_std = np.std(rmse_1d,axis=0)

rmse_37d = np.load('models/models_37dnorm/multi_test_result_chunk.npy')
rmse_37d_mean = np.mean(rmse_37d,axis=0)
rmse_37d_std = np.std(rmse_37d,axis=0)

room_list = ['A','B','C','D']
fig = plot_tools.plot_bar((rmse_1d_mean,rmse_1d_std),
                          (rmse_37d_mean,rmse_37d_std),
                          legend=['overal_norm','sep_norm'],
                          xticklabels=room_list,
                          xlabel='Room',ylabel=('RMSE($^o$)'),ylim=[0,5)
legend = fig.get_axes()[0].get_legend()
legend._set_loc(2)
plot_tools.savefig(fig,name='rmse_diff_norm.png')
