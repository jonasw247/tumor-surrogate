import numpy as np
from matplotlib import pyplot as plt
import torch

def crop(x, center_x, center_y, center_z):
    center_x = int(round(center_x.item() * 128))
    center_y = int(round(center_y.item() * 128))
    center_z = int(round(center_z.item() * 128))
    return x[center_x - 32:center_x + 32,
           center_y - 32:center_y + 32,
           center_z - 32:center_z + 32, :]

def get_gt_img(sample_name = '10_13_16'):
    data_path = f'/mnt/Drive3/ivan/data/valid/{sample_name}.npz'
    data = np.load(data_path)
    parameters = data['y']
    output = data['x'][:, :, :, 0:1]
    output = crop(output, parameters[3], parameters[4], parameters[5])
    return torch.tensor(output).permute(3,0,1,2)

if __name__ == '__main__':
    # sample 87
    # dice02: 0.9830, dice04: 0.9821, dice08: 0.9773, mae_wm: 0.0133162, mae_gm: 0.012050265, mae_csf: 0.00082050124
    # parameters: [2.30e-04 1.94e-02 1.60e+01 4.37e-01 5.36e-01 4.91e-01]
    sample_name = '10_13_16'
    data_path = f'/mnt/Drive3/ivan/data/valid/{sample_name}.npz'
    data = np.load(data_path)
    x = data['x'][:, :, :, 1:]
    parameters = data['y']
    output = data['x'][:, :, :, 0:1]

    plt.imshow(output[:, :, 64, 0])
    plt.show()

    x_025 = np.copy(output)
    x_07 = np.copy(output)
    x_025[x_025 >= 0.25] = 1
    x_025[x_025 < 0.25] = 0

    x_07[x_07 >= 0.7] = 1
    x_07[x_07 < 0.7] = 0

    plt.imshow(x_025[:, :, 64, 0])
    plt.show()

    plt.imshow(x_07[:, :, 64, 0])
    plt.show()


    np.savez('neural_inference/x_obs_test.npz', x=x, parameters=parameters, output=output, x_025=x_025, x_07=x_07)