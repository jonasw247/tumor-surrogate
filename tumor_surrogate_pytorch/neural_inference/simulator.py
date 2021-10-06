import glob

import numpy as np
import os
import torch

from tumor_surrogate_pytorch.model import TumorSurrogate


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


# def simulate_parameters_from_uniform(parameters):
#     parameters[0] = parameters[0] * 0.0007 + 0.0001
#     parameters[1] = parameters[1] * 0.0299 + 0.0001
#     parameters[2] = int(parameters[2] * 20 + 1)
#     parameters[6] = parameters[6] * 0.2 + 0.6
#     parameters[3] = parameters[3] * 0.5 + 0.25
#     parameters[4] = parameters[4] * 0.5 + 0.25
#     parameters[5] = parameters[5] * 0.5 + 0.25
#
#     parameters[7] = parameters[7] * 0.55 + 0.05
#     return parameters


class Simulator():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anatomy_dataset = BrainAnatomyDataset(device=self.device)

        self.net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])

        if os.path.exists(f'./tumor_surrogate_pytorch/saved_model/whole_dataset_orig_loss_best'):
            self.net = load_weights(self.net, path=f'./tumor_surrogate_pytorch/saved_model/whole_dataset_orig_loss_best')

        self.net = self.net.to(device=self.device)
        self.net.eval()

    def uncrop(self, x, center_x, center_y, center_z):
        out = torch.zeros([x.shape[0], 1, 128, 128, 128], device=self.device)
        for idx, sample in enumerate(x):
            center_x_curr = int(round(center_x[idx].item() * 128))
            center_y_curr = int(round(center_y[idx].item() * 128))
            center_z_curr = int(round(center_z[idx].item() * 128))

            out[idx, :, center_x_curr - 32:center_x_curr + 32,
            center_y_curr - 32:center_y_curr + 32,
            center_z_curr - 32:center_z_curr + 32] = x[idx]

        return out

    def predict_tumor_label_map(self, parameters):
        with torch.no_grad():
            if len(parameters.shape) == 1:
                parameters = parameters[None]
            parameters = parameters.to(self.device)
            parameters_net = parameters[:, 0:3].clone()
            parameters_net[:, 2] = torch.round(parameters_net[:, 2] * 20 + 1)
            y_range = [[0.0003, 0.0009], [0.0051, 0.0299], [0.0, 20.0]]
            for i, ri in enumerate(y_range):
                parameters_net[:,i] = (parameters_net[:,i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
                parameters_net[:,i] = torch.round(parameters_net[:,i] * 10 ** 2) / 10 ** 2
            #parameters = simulate_parameters_from_uniform(parameters)
            #parameters = torch.tensor(parameters).float()

            # random patient anatomy
            center_x = parameters[:,3]
            center_y = parameters[:,4]
            center_z = parameters[:,5]
            threshold07 = parameters[:,6]
            threshold025 = parameters[:,7]
            anatomy = self.anatomy_dataset.getitem(center_x=center_x,
                                                   center_y=center_y,
                                                   center_z=center_z)
            # call tumor simulator net and predict density
            anatomy = anatomy.to(self.device)

            output_batch, _ = self.net(anatomy, parameters_net)

            output_batch = self.uncrop(output_batch, center_x=center_x, center_y=center_y, center_z=center_z)

            # threshold at 0.25 and 0.7
            thresholded_025 = output_batch.clone()
            thresholded_07 = output_batch.clone()

            thresholded_025[thresholded_025 >= threshold025[:,None,None,None,None]] = 1
            thresholded_025[thresholded_025 < threshold025[:,None,None,None,None]] = 0

            thresholded_07[thresholded_07 >= threshold07[:,None,None,None,None]] = 1
            thresholded_07[thresholded_07 < threshold07[:,None,None,None,None]] = 0

            output = thresholded_025 + thresholded_07
            output = output.view(output.shape[0], -1)
            return output.cpu()

    def predict_tumor_density(self, parameters, brain_id=None):
        with torch.no_grad():
            if len(parameters.shape) == 1:
                parameters = parameters[None]
            parameters = parameters.to(self.device)
            parameters_net = parameters[:, 0:3].clone()
            parameters_net[:, 2] = torch.round(parameters_net[:, 2] * 20 + 1)
            y_range = [[0.0003, 0.0009], [0.0051, 0.0299], [0.0, 20.0]]
            for i, ri in enumerate(y_range):
                parameters_net[:,i] = (parameters_net[:,i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
                parameters_net[:,i] = torch.round(parameters_net[:,i] * 10 ** 2) / 10 ** 2

            # random patient anatomy
            center_x = parameters[:,3]
            center_y = parameters[:,4]
            center_z = parameters[:,5]
            threshold07 = parameters[:,6]
            threshold025 = parameters[:,7]
            anatomy = self.anatomy_dataset.getitem(center_x=center_x,
                                                   center_y=center_y,
                                                   center_z=center_z,
                                                   brain_id=brain_id)
            # call tumor simulator net and predict density
            anatomy = anatomy.to(self.device)

            output_batch, _ = self.net(anatomy, parameters_net)

            return output_batch.sum(dim=0).cpu()




class BrainAnatomyDataset:
    def __init__(self, device):
        data_path = '/home/ivan_nas/npe_data/train/'
        self.samples = sorted(glob.glob(data_path + '*'))
        self.brains = [torch.from_numpy(np.load(sample)).permute((3, 0, 1, 2)) for sample in self.samples]
        self.device = device

    def crop(self, x, center_x, center_y, center_z):
        center_x = int(round(center_x.item() * 128))
        center_y = int(round(center_y.item() * 128))
        center_z = int(round(center_z.item() * 128))
        return x[:,center_x - 32:center_x + 32,
               center_y - 32:center_y + 32,
               center_z - 32:center_z + 32]

    def getitem(self, center_x, center_y, center_z, brain_id=None):

        batch_size = center_x.shape[0]
        if brain_id:
            data = torch.empty((batch_size, 3, 64, 64, 64), device=self.device)
            data_path = f'/home/marc_chan/data/data/valid/{brain_id}.npz'
            brain = np.load(data_path)
            brain = torch.from_numpy(brain['x'][:, :, :, 1:]).permute((3, 0, 1, 2))
            for i in range(batch_size):
                data[i] = self.crop(brain, center_x[i], center_y[i], center_z[i])
            return data

        else:
            idxes = np.random.randint(low=0, high=10, size=batch_size)
            data = torch.empty((batch_size, 3, 64, 64, 64), device=self.device)
            for i, idx in enumerate(idxes):
                cropped_data = self.crop(self.brains[idx], center_x[i], center_y[i], center_z[i])
                data[i] = cropped_data

            return data


if __name__ == '__main__':
    ds = BrainAnatomyDataset()
    item = ds.getitem(0.5, 0.5, 0.5)

    sim = Simulator()
    parameters = np.array([2.30e-04, 1.94e-02, 1.60e+01, 4.37e-01, 5.36e-01, 4.91e-01, 0.25, 0.7])
    # 0.00023, 0.019, 16, 0.43, 0.53, 0.49
    sim.predict_tumor_label_map(parameters)
    a = 1
