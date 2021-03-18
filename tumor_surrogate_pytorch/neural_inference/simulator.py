import glob

import numpy as np
import os
import torch

from tumor_surrogate_pytorch.model import TumorSurrogate


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def simulate_parameters_from_uniform(parameters):
    parameters[0] = parameters[0] * 0.0007 + 0.0001
    parameters[1] = parameters[1] * 0.0299 + 0.0001
    parameters[2] = int(parameters[2] * 20 + 1)
    parameters[6] = parameters[6] * 0.2 + 0.6
    parameters[3] = parameters[3] * 0.5 + 0.25
    parameters[4] = parameters[4] * 0.5 + 0.25
    parameters[5] = parameters[5] * 0.5 + 0.25

    parameters[7] = parameters[7] * 0.55 + 0.05
    return parameters


class Simulator():
    def __init__(self):
        self.anatomy_dataset = BrainAnatomyDataset()

        self.net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
        os.environ['CUDA_VISIBLE_DEVICES'] = "7"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(f'./tumor_surrogate_pytorch/saved_model/whole_dataset_orig_loss_best'):
            self.net = load_weights(self.net, path=f'./tumor_surrogate_pytorch/saved_model/whole_dataset_orig_loss_best')

        self.net = self.net.to(device=self.device)
        self.net.eval()

    def uncrop(self, x, center_x, center_y, center_z):
        center_x = int(round(center_x.item() * 128))
        center_y = int(round(center_y.item() * 128))
        center_z = int(round(center_z.item() * 128))

        out = torch.zeros([1, 1, 128, 128, 128])
        out[:, :, center_x - 32:center_x + 32,
        center_y - 32:center_y + 32,
        center_z - 32:center_z + 32] = x

        return out

    def predict_tumor_density(self, parameters):
        parameters = simulate_parameters_from_uniform(parameters)
        # random patient anatomy
        center_x = parameters[3]
        center_y = parameters[4]
        center_z = parameters[5]
        threshold07 = parameters[6]
        threshold025 = parameters[7]
        anatomy = self.anatomy_dataset.getitem(center_x=center_x,
                                               center_y=center_y,
                                               center_z=center_z)
        # call tumor simulator net and predict density
        parameters = torch.tensor(parameters).float()
        anatomy, parameters = anatomy.to(self.device), parameters.to(self.device)
        anatomy = anatomy[None, :]
        parameters = parameters[None, 0:3]
        with torch.no_grad():
            output_batch, _ = self.net(anatomy, parameters[None, 0:3])

        output_batch = self.uncrop(output_batch, center_x=center_x, center_y=center_y, center_z=center_z)

        # threshold at 0.25 and 0.7
        thresholded_025 = output_batch.clone()
        thresholded_07 = output_batch.clone()

        thresholded_025[thresholded_025 >= threshold025] = 1
        thresholded_025[thresholded_025 < threshold025] = 0

        thresholded_07[thresholded_07 >= threshold07] = 1
        thresholded_07[thresholded_07 < threshold07] = 0

        output = thresholded_025 + thresholded_07
        output = output.flatten()
        return output


class BrainAnatomyDataset:
    def __init__(self):
        data_path = '/home/ivan_nas/npe_data/train/'
        self.samples = sorted(glob.glob(data_path + '*'))

    def crop(self, x, center_x, center_y, center_z):
        center_x = int(round(center_x.item() * 128))
        center_y = int(round(center_y.item() * 128))
        center_z = int(round(center_z.item() * 128))
        return x[center_x - 32:center_x + 32,
               center_y - 32:center_y + 32,
               center_z - 32:center_z + 32]

    def getitem(self, center_x, center_y, center_z):
        idx = np.random.randint(low=0, high=10)
        data = np.load(self.samples[idx])
        data = self.crop(data, center_x, center_y, center_z)
        data = torch.tensor(data).permute((3, 0, 1, 2)).float()

        return data


if __name__ == '__main__':
    ds = BrainAnatomyDataset()
    item = ds.getitem(0.5, 0.5, 0.5)

    sim = Simulator()
    parameters = np.array([2.30e-04, 1.94e-02, 1.60e+01, 4.37e-01, 5.36e-01, 4.91e-01, 0.25, 0.7])
    # 0.00023, 0.019, 16, 0.43, 0.53, 0.49
    sim.predict_tumor_density(parameters)
    a = 1
