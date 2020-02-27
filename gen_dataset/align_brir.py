import os
import numpy as np
import pysofa
from BasicTools import DspTools
import matplotlib.pyplot as plt

"""Align brirs of reverberant rooms to that from the anechoic room
"""

brirs_dir = os.path.expanduser('~/Work_Space/Data/RealRoomBRIRs')
brirs_aligned_dir = 'brirs_aligned'
os.mkdir(brirs_aligned_dir, exist_ok=True)

n_azi = 37
n_channel = 2
rever_room_all = ('Room_A', 'Room_B', 'Room_C', 'Room_D')
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']



def plot_brirs():
    for room in room_all:
        brirs_fpath = f'brirs_aligned/{room}.npy'
        brirs = np.load(brirs_fpath)
        print(brirs.shape)
        fig, ax = plt.subplots(1, 1)
        for azi_i in range(0, 37, 8):
            ax.plot(brirs[azi_i, :, 0])
        ax.set_xlim([0, 2000])
        fig.savefig(f'brirs_aligned/{room}.png')
        plt.close(fig)


def load_brirs(room):
    brirs_fpath = os.path.join(brirs_dir, f'UniS_{room}_BRIR_16k.sofa')
    brirs = pysofa.SOFA(brirs_fpath).FIR.IR.transpose(0, 2, 1)
    return brirs


def align_brirs():
    """
    For each reverberant room, calculate BRIRs delays of each sound position
    and align BRIRs according to the averaged delay
    """

    brirs_anechoic = load_brirs('Anechoic')
    np.save(os.path.join(brirs_aligned_dir, 'Anechoic.npy'), brirs_anechoic)

    delay_all = np.zeros((n_azi, n_channel))
    for reverb_room in rever_room_all:
        print(reverb_room)
        brirs = load_brirs(reverb_room)
        for azi_i in range(n_azi):
            for channel_i in range(n_channel):
                delay_all[azi_i, channel_i] = DspTools.cal_delay(
                                            brirs_anechoic[azi_i, :, channel_i],
                                            brirs[azi_i, :, channel_i])
        delay_mean = np.int16(np.round(np.mean(delay_all)))
        if delay_mean > 0:
            brirs_aligned = np.concatenate((np.zeros((n_azi, delay_mean,
                                                      n_channel)),
                                            brirs),
                                           axis=1)
        else:
            brirs_aligned = brirs[:, -delay_mean:, :]

        np.save(os.path.join(brirs_aligned_dir, f'{reverb_room}.npy'),
                brirs_aligned)

        if False:
            fig, ax = plt.subplots(1, 1)
            ax.plot(brirs_anechoic[0, :, 0], label='anechoic')
            ax.plot(brirs_aligned[0, :, 0], label='reverb_aligned')
            ax.plot(brirs[0, :, 0], label='reverb')
            ax.set_xlim((0, 500))
            ax.legend()
            ax.set_title(reverb_room)
            fig.savefig(os.path.join(brirs_aligned_dir,
                                     f'{reverb_room}_aligned.png'))
            plt.close(fig)


if __name__ == '__main__':
    align_brirs()
    # plot_brirs()
