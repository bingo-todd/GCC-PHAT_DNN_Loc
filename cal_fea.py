import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from multiprocessing import Pool

home_dir = os.path.expanduser('~')
sys.path.append(os.path.join(home_dir,
                'Work_Space/my_module/basic_tools/basic_tools'))
import gcc
import process_bar
import plot_tools
import wav_tools
from get_fpath import get_fpath

def gcc_phat_parallel_f(frame_binaural):
    return gcc.cal_gcc_phat(frame_binaural[:,0],frame_binaural[:,1],
                            max_delay=18)

def cal_fea(record_dir,fea_dir):
    """calculate GCC-PHAT features
    Args:
        record_dir: wave dataset directory
    """
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    wav_fpath_all = get_fpath(dir=record_dir,suffix='.wav',
                              pattern='(reverb)',is_exclude=False)
    n_file = len(wav_fpath_all)


    pb = process_bar.process_bar(max_value=n_file,title='GCC_PHAT {}'.format(record_dir))
    parallel_pool = Pool(6)

    for wav_fpath in wav_fpath_all:
        data,fs = wav_tools.wav_read(os.path.join(record_dir,wav_fpath))
        frame_all = wav_tools.frame_data(data,frame_len=320,shift_len=160)
        n_frame = frame_all.shape[0]

        #
        # fea = np.asarray([gcc.cal_gcc_phat(frame_all[i,:,0],
        #                                    frame_all[i,:,1],
        #                                    max_delay=18)
        #                              for i in range(n_frame)])

        # parallel version
        fea_frame_all = parallel_pool.map(gcc_phat_parallel_f,
                                          [frame_all[i] for i in range(n_frame)])
        fea_frame_all = np.asarray(fea_frame_all)

        fea_fpath = os.path.join(fea_dir,
                                 ''.join((wav_fpath[:-4],'.npy')))
        dir_tmp = os.path.dirname(fea_fpath)
        if not os.path.exists(dir_tmp):
            os.makedirs(dir_tmp)
        np.save(fea_fpath,fea_frame_all)

        pb.update()


def file_reader(data_dir):
    """Read spectrum files under given directory
    Args:
        data_dir:
    Returns:
        sample generator, [samples,label_onehots]
    """
    if isinstance(data_dir,list):
        dir_list = data_dir
    else:
        dir_list = [data_dir]

    for sub_set_dir in dir_list:
        fea_fpath_list = get_fpath(sub_set_dir,'.npy')
        for fea_fpath in fea_fpath_list:
            fea_fpath_abs = os.path.join(sub_set_dir,fea_fpath)
            fea = np.load(fea_fpath_abs)
            yield fea



def cal_norm_params(fea_dir):
    """
    """
    mean_37d = 0
    n_sample = 0
    for samples in file_reader(fea_dir):
        mean_37d = mean_37d + np.sum(samples,axis=0)
        n_sample = n_sample+samples.shape[0]
    mean_37d = mean_37d/n_sample
    mean_1d = np.mean(mean_37d)

    std_37d = 0
    std_1d = 0
    for samples in file_reader(fea_dir):
        std_37d = std_37d + np.sum((samples-mean_37d)**2,axis=0)
        std_1d = std_1d + np.sum((samples-mean_1d)**2)
    std_37d = np.sqrt(std_37d/n_sample)
    std_1d = np.sqrt(std_1d/n_sample)

    return [[mean_1d,std_1d],[mean_37d,std_37d]]




if __name__ == "__main__":

    # for i in range(1,5):
    #     print(i)
    #     cal_fea(record_dir='/mnt/hd6t/songtao/Localize/WavLoc/Data/v{}'.format(i),
    #           fea_dir='/mnt/hd6t/songtao/Localize/WavLoc/Data/v{}/GCC_PHAT'.format(i))
    #

    # calculate the normalization params of mct training dataset
    room_list = ['Anechoic','A','B','C','D']
    dataset_dir = '/mnt/hd6t/songtao/Localize/WavLoc/Data/v1/GCC_PHAT/train/reverb'
    norm_param_dir = 'norm_params'

    fig,ax = plt.subplots(1,1)
    tau = np.arange(-18,19)
    bar_width = 0.2
    for room_i,tar_room in enumerate(room_list[1:]):
        mct_filter = lambda room : room != tar_room
        room_dataset_dirs=[os.path.join(dataset_dir,room)
        					 for room in filter(mct_filter,room_list)]
        [[mean_1d,std_1d],
         [mean_37d,std_37d]]=cal_norm_params(room_dataset_dirs)

        param_name = 'mct_not_{}'.format(tar_room)
        np.save(os.path.join(norm_param_dir,'{}_37d.npy'.format(param_name)),
                [mean_37d,std_37d])

        np.save(os.path.join(norm_param_dir,'{}_1d.npy'.format(param_name)),
                [mean_1d,std_1d])

        ax.errorbar(x=tau+(room_i-1)*bar_width,y=mean_37d,yerr=std_37d,label=tar_room)
    ax.set_xlabel('Delay(sample)')
    ax.legend()
    plot_tools.savefig(fig,name='norm_params.png')
    #
    #
    # # normalization example
    # # example()
