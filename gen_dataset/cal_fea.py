import numpy as np
from multiprocessing import Pool
import os
from BasicTools import wav_tools, get_file_path, ProgressBar, gcc


data_dir_base = '../../Data'
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

    wav_fpath_all = get_file_path(dir_path=record_dir, suffix='.wav',
                                  pattern='reverb')

    pb = ProgressBar(max_value=len(wav_fpath_all),
                     title=f'GCC_PHAT {record_dir}')
    pool = Pool(24)
    for wav_fpath in wav_fpath_all:
        fea_fpath = os.path.join(fea_dir, '{}.npy'.format(wav_fpath[:-4]))
        if os.path.exists(fea_fpath):
            # warnings.warn(f'{fea_fpath} exists!')
            continue

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


if __name__ == "__main__":

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
