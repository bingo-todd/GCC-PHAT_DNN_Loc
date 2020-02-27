import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import warnings
import os
from BasicTools import plot_tools, wav_tools, get_fpath, ProcessBar, gcc


data_dir_base = '../Data'
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
max_delay = 18


def gcc_phat_parallel_f(frame_binaural):
    return gcc.cal_gcc_phat(frame_binaural[:, 0], frame_binaural[:, 1],
                            max_delay=max_delay)


def cal_fea(record_dir, fea_dir):
    """calculate GCC-PHAT features
    Args:
        record_dir: wave dataset directory
    """
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    wav_fpath_all = get_fpath(dir_path=record_dir, suffix='.wav',
                              pattern='reverb/Room_D')

    pb = ProcessBar(max_value=len(wav_fpath_all),
                    title=f'GCC_PHAT {record_dir}')
    pool = Pool(24)
    for wav_fpath in wav_fpath_all:
        fea_fpath = os.path.join(fea_dir, '{}.npy'.format(wav_fpath[:-4]))
        # if os.path.exists(fea_fpath):
        #     warnings.warn(f'{fea_fpath} exists!')
        #    continue

        data, fs = wav_tools.read_wav(os.path.join(record_dir, wav_fpath))
        frame_all = wav_tools.frame_data(data, frame_len=320, shift_len=160)
        n_frame = frame_all.shape[0]

        fea_frame_all = pool.map(gcc_phat_parallel_f,
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

    if True:
        print('v1 train')
        cal_fea(record_dir=os.path.join(data_dir_base, 'v1/train'),
                fea_dir=os.path.join(data_dir_base, 'v1/GCC_PHAT/train'))

        print('v1 valid')
        cal_fea(record_dir=os.path.join(data_dir_base, 'v1/valid'),
                fea_dir=os.path.join(data_dir_base, 'v1/GCC_PHAT/valid'))

        for i in range(1, 5):
            print(f'v{i} test')
            cal_fea(record_dir=os.path.join(data_dir_base, f'v{i}/test'),
                    fea_dir=os.path.join(data_dir_base, f'v{i}/GCC_PHAT/test'))
    if False:
        # calculate the normalization params of mct training dataset
        dataset_dir = os.path.join(data_dir_base, 'v1/GCC_PHAT/train/reverb')
        norm_param_dir = 'norm_params'

        fig, ax = plt.subplots(1, 1)
        tau = np.arange(-18, 19)
        bar_width = 0.2
        for room_i, room_tar in enumerate(room_all[1:]):
            mct_room_all = [room for room in room_all if room != room_tar]

            dataset_room_dir_all = [os.path.join(dataset_dir, room)
                                    for room in mct_room_all]
            [[mean_1d, std_1d],
            [mean_37d, std_37d]] = cal_norm_params(dataset_room_dir_all)

            param_name = 'mct_{}'.format(room_tar)
            np.save(os.path.join(norm_param_dir, '{}_37d.npy'.format(param_name)),
                    [mean_37d, std_37d])

            np.save(os.path.join(norm_param_dir, '{}_1d.npy'.format(param_name)),
                    [mean_1d, std_1d])

            ax.errorbar(x=tau+(room_i-1)*bar_width, y=mean_37d, yerr=std_37d,
                        label=room_tar)
        ax.set_xlabel('Delay(sample)')
        ax.legend()
        plot_tools.savefig(fig, fig_name='norm_params.png')

        # normalization parameters of all train data
        if True:
            dataset_room_dir_all = [os.path.join(dataset_dir, room)
                                    for room in room_all]
            [[mean_1d, std_1d],
            [mean_37d, std_37d]] = cal_norm_params(dataset_room_dir_all)

            np.save(os.path.join(norm_param_dir, 'all_room_37d.npy'),
                    [mean_37d, std_37d])

            np.save(os.path.join(norm_param_dir, 'all_room_1d.npy'),
                    [mean_1d, std_1d])

            tau = np.arange(-18, 19)
            fig, ax = plt.subplots(1, 1)
            ax.errorbar(x=tau, y=mean_37d, yerr=std_37d)
            ax.set_xlabel('Delay(sample)')
            plot_tools.savefig(fig, fig_name='norm_params.png')
