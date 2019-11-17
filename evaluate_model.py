import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import LocDNN
workspace_dir = os.path.join(os.path.expanduser('~'),'Work_Space')
sys.path.append(os.path.join(workspace_dir,'my_module/basic_tools/basic_tools'))
import plot_tools


def evaluate_mct(model_dir,dataset_dir,chunk_size=25):
    reverb_rooms = ['A','B','C','D']
    n_room = len(reverb_rooms)

    rmse_mct = np.zeros(n_room) #
    for room_i,room in enumerate(reverb_rooms):
        config_fpath = os.path.join(model_dir,
                                    'mct_not_{}/'.format(room),
                                    'mct_not_{}.cfg'.format(room))
        model = LocDNN.LocDNN(config_fpath=config_fpath,
                              is_load_model=True,
                              gpu_index=1)
        rmse_mct[room_i] = model.evaluate_chunk(os.path.join(dataset_dir,room),
                                                chunk_size=chunk_size)
    return rmse_mct



if __name__ == '__main__':
    chunk_size = 25
    model_dir = sys.argv[1]
    rmse_multi_test = np.zeros((4,4))
    for test_i in range(1,5):
        test_set_dir = os.path.join('/mnt/hd6t/songtao/Localize/WavLoc/Data',
                                    'v{}/GCC_PHAT/test/reverb'.format(test_i))
        rmse_multi_test[test_i-1,:] = evaluate_mct(model_dir=model_dir,
                                                   dataset_dir=test_set_dir,
                                                   chunk_size=chunk_size)

    np.save(os.path.join(model_dir,'multi_test_result_chunk.npy'),
            rmse_multi_test)

    rmse_mean = np.mean(rmse_multi_test,axis=0)
    rmse_std = np.std(rmse_multi_test,axis=0)
    print('mean: {} \n std: {}'.format(rmse_mean,rmse_std))
    with open(os.path.join(model_dir,'multi_test_result_chunk.txt'),'w') as result_file:
        result_file.write('multi_test: {}\n'.format(rmse_multi_test))
        result_file.write('mean: {}\n'.format(rmse_mean))
        result_file.write('std: {}\n'.format(rmse_std))
