from pathlib import Path

import os
import torch
from torch.utils.data import DataLoader

from tumor_surrogate_pytorch.config import get_config
from tumor_surrogate_pytorch.data import TumorDataset
from tumor_surrogate_pytorch.model import TumorSurrogate
from tumor_surrogate_pytorch.utils import create_hists


def load_weights(model, path):
    model.load_state_dict(torch.load(path)['state_dict'])
    model.eval()
    return model


def test(data_path, run_name):
    net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device)

    if os.path.exists(f'./tumor_surrogate_pytorch/saved_model/{run_name}'):
        net = load_weights(net, path=f'./tumor_surrogate_pytorch/saved_model/{run_name}')
    else:
        raise Exception(f'No trained model exists at ./tumor_surrogate_pytorch/saved_model/{run_name}. Please train a model before testing.')
    net.to(device=device)

    save_path = f'./tumor_surrogate_pytorch/test_output/{run_name}/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    dataset = TumorDataset(data_path=data_path, dataset='valid/')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=32)
    create_hists(net, loader, device, save_path)


if __name__ == '__main__':
    config, unparsed = get_config()
    test(config.data_path, config.run_name)
