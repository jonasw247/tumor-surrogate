import glob

import numpy as np
import torch

if __name__ == '__main__':
    save_path = '/home/marc_chan/data/brains/'
    data_path = '/home/marc_chan/data/tumor_mparam/v/'

    samples = sorted(glob.glob(save_path + '*'))
    brains = [torch.from_numpy(np.load(sample)).permute((3, 0, 1, 2)) for sample in samples]


    for i in range(0,10):
        file_name = f'{i}_0_1.npz'
        brain = np.load(data_path+file_name)
        brain = torch.from_numpy(brain['x'][:, :, :, 1:])
        np.save(save_path+f'brain_{i}.npy', brain)