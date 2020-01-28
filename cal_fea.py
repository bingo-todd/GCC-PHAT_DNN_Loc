import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import warnings
import os
import sys
my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import gcc  # noqa: E402
import plot_tools  # noqa: E402
import wav_tools  # noqa: E402
from get_fpath import get_fpath  # noqa: E402
from ProcessBar import ProcessBar  # noqa: E402


def gcc_phat_parallel(frame_binaural):
    return gcc.cal_gcc_phat(frame_binaural[:, 0], frame_binaural[:, 1],
                            max_delay=18)


def cal_fea(record_dir, fea_dir):
    """calculate GCC-PHAT features
    Args:
        record_dir: wave dataset directory
    """
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    wav_fpath_all = get_fpath(dir=record_dir, suffix='.wav',
                              pattern='(reverb)')

    pb = ProcessBar(max_value=len(wav_fpath_all),
                    title=f'GCC_PHAT {record_dir}')
    pool = Pool(24)
    for wav_fpath in wav_fpath_all:
        fea_fpath = os.path.join(fea_dir, '{}.npy'.format(wav_fpath[:-4]))
        if os.path.exists(fea_fpath):
            warnings.warn(f'{fea_fpath} exists!')
            continue

        record, fs = wav_tools.wav_read(os.path.join(record_dir, wav_fpath))
        frame_all = wav_tools.frame_data(record, frame_len=320, shift_len=160)
        n_frame = frame_all.shape[0]

        fea_frame_all = pool.map(gcc_phat_parallel,
                                 [frame_all[i] for i in range(n_frame)])
        fea_frame_all = np.asarray(fea_frame_all)

        dir_tmp = os.path.dirname(fea_fpath)
        if not os.path.exists(dir_tmp):
            os.makedirs(dir_tmp)
        np.save(fea_fpath, fea_frame_all)

        pb.update()


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
        fea_fpath_list = get_fpath(sub_set_dir, '.npy')
        for fea_fpath in fea_fpath_list:
            fea_fpath_abs = os.path.join(sub_set_dir, fea_fpath)
            fea = np.load(fea_fpath_abs)
            yield fea


def cal_norm_params(fea_dir):
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

    data_dir = '../../WaveLoc/Data'

    if False:
        cal_fea(record_dir='../../Data/v1/train',
                fea_dir='../../Data/v1/GCC_PHAT_hann/train')
        cal_fea(record_dir='../../Data/v1/valid',
                fea_dir='../../Data/v1/GCC_PHAT_hann/valid')

        for i in range(1, 5):
            cal_fea(record_dir=f'../../Datav{i}/test',
                    fea_dir=f'../../Data/v{i}/GCC_PHAT_hann/test')

    if True:

        # calculate the normalization params of mct training dataset
        room_all = ['Anechoic', 'A', 'B', 'C', 'D']
        reverb_room_all = ['A', 'B', 'C', 'D']
        data_dir = '../../Data/v1/GCC_PHAT/train'
        norm_param_dir = 'norm_params'

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

            np.save(os.path.join(norm_param_dir, '37d', f'mct_{room_tar}.npy'),
                    [mean_37d, std_37d])

            np.save(os.path.join(norm_param_dir, '1d', f'mct_{room_tar}.npy'),
                    [mean_1d, std_1d])

            ax.errorbar(x=tau+(room_i-1)*bar_width, y=mean_37d, yerr=std_37d,
                        label=room_tar)
        ax.errorbar(x=tau, y=mean_37d, yerr=std_37d)
        ax.set_xlabel('Delay(sample)')
        plot_tools.savefig(fig, fig_name='norm_params_mct.png')
