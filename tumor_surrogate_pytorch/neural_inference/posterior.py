from pathlib import Path

import GPUtil
import dill as dill
import numpy as np
import os
import torch
import torchvision
import pickle
from sbi import utils
from sbi.analysis import eval_conditional_density
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT
from tumor_surrogate_pytorch.neural_inference.create_observation import get_gt_img

from tumor_surrogate_pytorch.neural_inference.embedding_net import ConvNet
from tumor_surrogate_pytorch.neural_inference.simulator import Simulator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import argparse
import warnings
warnings.filterwarnings("ignore")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

arg_lists = []
parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--num_simulations', type=int, default=5000)
parser.add_argument('--start_round', type=int, default=None)

parser.add_argument('--run_name', type=str, default='debug')
parser.add_argument('--gpu', type=str, default='0')

param_to_name = {
    0: 'D',
    1: '\u03C1',
    2: 'T',
    3: 'x',
    4: 'y',
    5: 'z',
    6: 'u_1',
    7: 'u_2'
}

def print_gpu_utilisation():
    gpu = GPUtil.getGPUs()[1]
    print(f'{gpu.memoryUsed} MB allocated\n')

def plot_probabilities(posterior, ranges, gts):
    steps = 100
    fig, axs = plt.subplots(2, 4, sharey=True, tight_layout=True, figsize=(15,5))
    for i, r in enumerate(ranges):
        x = np.linspace(r[0] + 1e-32, r[1] - 1e-32, steps)
        probs = eval_conditional_density(posterior, torch.tensor(gts, device=torch.device('cuda')),
                                         torch.tensor(ranges, device=torch.device('cuda')), dim1=i, dim2=i, resolution=steps)
        if i == 2:
            x = x * 20 + 1
            axs[i//4, i % 4].axvline(x=gts[i] * 20 + 1, linestyle='dotted', color='red')
        else:
            axs[i//4, i % 4].axvline(x=gts[i], linestyle='dotted', color='red')


        axs[i//4, i % 4].locator_params(axis="x", nbins=5)
        axs[i//4, i % 4].plot(x, probs.cpu().numpy())
        axs[i//4, i % 4].set_title(f"Parameter {param_to_name[i]}")
    plt.savefig(f'tumor_surrogate_pytorch/neural_inference/output/run2_noreuse/plots/round_2_parameters.png')
    #plt.show()
    plt.clf()
    plt.close()


class NPE:
    def __init__(self, simulator, device, log_path, run_name, gts, ranges):
        self.device = device
        self.prior = utils.BoxUniform(low=torch.tensor([0.0001, 0.0001, 0, 0.4, 0.4, 0.4, 0.6, 0.05], device=device),
                                      high=torch.tensor([0.0008, 0.03, 1, 0.6, 0.6, 0.6, 0.8, 0.6], device=device))
        self.simulator = simulator
        self.posteriors = []
        self.writer = SummaryWriter(log_dir=log_path+run_name)
        self.run_name = run_name
        self.gts = gts
        self.ranges = ranges
        Path(f'tumor_surrogate_pytorch/neural_inference/output/{run_name}/state').mkdir(parents=True, exist_ok=True)
        Path(f'tumor_surrogate_pytorch/neural_inference/output/{run_name}/posterior').mkdir(parents=True, exist_ok=True)
        Path(f'tumor_surrogate_pytorch/neural_inference/output/{run_name}/inference').mkdir(parents=True, exist_ok=True)
        Path(f'tumor_surrogate_pytorch/neural_inference/output/{run_name}/plots').mkdir(parents=True, exist_ok=True)

    def save_inference(self, inference, path):
        with open(path, "wb") as handle:
            dill.dump(inference, handle)

    def load_inference(self, path):
        with open(path, "rb") as handle:
            inference = dill.load(handle)
        return inference

    def save_posterior(self, posterior, path):
        with open(path, "wb") as handle:
            pickle.dump(posterior, handle)

    def load_posterior(self, path):
        with open(path, "rb") as handle:
            posterior = pickle.load(handle)
        return posterior

    def save_state(self, inference, posterior, round, num_simulations):
        inference_path = f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/inference/round_{round}.pkl'
        posterior_path = f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/posterior/round_{round}.pkl'
        save_dict = {
            'round': round,
            'num_simulations': num_simulations,
            'inference_path': f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/inference/round_{round}.pkl',
            'posterior_path': f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/posterior/round_{round}.pkl'
        }
        self.save_inference(inference,inference_path)
        self.save_posterior(posterior, posterior_path)
        torch.save(save_dict, f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/state/round_{round}')

    def load_state(self, round):
        state_path = f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/state/round_{round}'
        load_dict = torch.load(state_path)
        round = load_dict['round']+1
        num_simulations = load_dict['num_simulations']
        inference_path = load_dict['inference_path']
        inference = self.load_inference(inference_path)
        inference._summary_writer = inference._default_summary_writer()
        print("Loading inference from: ", inference_path)

        posterior_path = load_dict['posterior_path']
        posterior = self.load_posterior(posterior_path)
        print("Loading posterior from: ", posterior_path)

        return inference, posterior, round, num_simulations

    def plot_probabilities(self, posterior, round):
        steps = 100
        for i, r in enumerate(self.ranges):
            x = np.linspace(r[0] + 1e-32, r[1] - 1e-32, steps)
            probs = eval_conditional_density(posterior, torch.tensor(self.gts, device=self.device),
                                             torch.tensor(self.ranges, device=self.device), dim1=i, dim2=i, resolution=steps)
            # plt.locator_params(axis="x", nbins=10)
            if i == 2:
                x = x * 20 + 1
                plt.axvline(x=self.gts[i] * 20 + 1, linestyle='dotted', color='red')
            else:
                plt.axvline(x=self.gts[i], linestyle='dotted', color='red')
            plt.plot(x, probs.cpu().numpy())
            plt.savefig(f'tumor_surrogate_pytorch/neural_inference/output/{self.run_name}/plots/round_{round}_paramerter_{i}.png')
            plt.clf()
            plt.close()

    def bayesian_inference(self, num_samples, proposal):
        thetas = proposal.sample((num_samples,))
        tumor_density = torch.zeros(1,64,64,64)
        for i in range(thetas.shape[0]//32+1):
            curr_theta = thetas[i*32:(i+1)*32]
            tumor_density += self.simulator.predict_tumor_density(curr_theta)

        tumor_density /= num_samples
        return tumor_density

    def forward(self, x_ob, num_rounds, num_simulations, start_round=None):
        print("Starting forward")
        neural_posterior = utils.posterior_nn(model='mdn',
                                              embedding_net=ConvNet(device=self.device), z_score_x=False)
        inference = APT(prior=self.prior, device='gpu', density_estimator=neural_posterior)
        simulator, prior = prepare_for_sbi(self.simulator.predict_tumor_label_map, self.prior)
        proposal = prior

        gt = get_gt_img(sample_name = '10_13_16')
        self.writer.add_image('ground truth', gt[0:1, :, :, 32], global_step=0)

        round = 0
        if start_round is not None:
            inference, posterior, round, num_simulations = self.load_state(start_round)
            proposal = posterior.set_default_x(x_ob)

        for i in range(round, num_rounds):
            print("Round: ", i)
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, simulation_batch_size=32)
            #x = torch.load('tumor_surrogate_pytorch/neural_inference/error/x25_04_2021_14_33_35.pt', map_location='cpu')
            #theta = torch.load('tumor_surrogate_pytorch/neural_inference/error/theta_old25_04_2021_14_33_35.pt', map_location='cpu')
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(show_train_summary=True, training_batch_size=32,
                                                                                                discard_prior_samples=True)
            sample_with_mcmc = False  # True if i < 2 else False
            posterior = inference.build_posterior(density_estimator, sample_with_mcmc=sample_with_mcmc,
                                                  rejection_sampling_parameters={'max_sampling_batch_size': 50})

            self.posteriors.append(posterior)
            self.save_state(inference, posterior, i, num_simulations)

            proposal = posterior.set_default_x(x_ob)
            self.plot_probabilities(proposal, i)
            tumor_density = self.bayesian_inference(1000, proposal)
            img = tumor_density[0:1,:,:,32]
            img.clamp_(min=img.min(), max=img.max())
            img.sub_(img.min()).div_(max(img.max() - img.min(), 1e-5))
            self.writer.add_image('bayesian plot', img, global_step=i)
            #map_estimate = proposal.map(num_init_samples=50, num_to_optimize=25, show_progress_bars=False)
            #for j in range(8):
            #    self.writer.add_scalar(f'parameter {j+1} map', map_estimate[j], global_step=i)
        return posterior


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    simulator = Simulator()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    npe = NPE(simulator=simulator, device=device, log_path='tumor_surrogate_pytorch/neural_inference/logs/', run_name=args.run_name,
              gts= [2.30e-04, 1.94e-02, 0.75, 4.37e-01, 5.36e-01, 4.91e-01, 0.7, 0.25],
              ranges=[[0.0001,0.0008], [0.0001,0.03], [0,1], [0.4,0.6], [0.4,0.6], [0.4,0.6], [0.6,0.8], [0.05, 0.6]])

    x_ob = np.load('tumor_surrogate_pytorch/neural_inference/x_obs_test.npz')
    x_ob = x_ob['x_025'] + x_ob['x_07']
    x_ob_img = np.copy(x_ob)
    x_ob = x_ob.flatten()
    x_ob = torch.tensor(x_ob, device=device)
    posterior = npe.forward(x_ob=x_ob, num_rounds=args.rounds, num_simulations=args.num_simulations, start_round=args.start_round)

    """
    with open('tumor_surrogate_pytorch/neural_inference/output/run1/posterior/round_4.pkl', "rb") as handle:
        posterior = pickle.load(handle)
        proposal = posterior.set_default_x(x_ob)
        simulator = Simulator()
        gt = get_gt_img(sample_name='10_13_16')

        map_estimate = proposal.map(num_init_samples=50, num_to_optimize=25, show_progress_bars=True)
        tumor_density = simulator.predict_tumor_density(map_estimate, brain_id='10_13_16')
        img = tumor_density[0, :, :, 32]
        img.clamp_(min=img.min(), max=img.max())
        img.sub_(img.min()).div_(max(img.max() - img.min(), 1e-5))
        #img = img.permute(1,2,0).cpu().numpy()

        thresholded_u1 = np.copy(img)
        thresholded_u2 = np.copy(img)
        thresholded_u1[thresholded_u1 >= map_estimate[6].cpu().numpy()] = 1
        thresholded_u1[thresholded_u1 < map_estimate[6].cpu().numpy()] = 0

        thresholded_u2[thresholded_u2 >= map_estimate[7].cpu().numpy()] = 1
        thresholded_u2[thresholded_u2 < map_estimate[7].cpu().numpy()] = 0

        fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True)
        axs[0,0].imshow(gt[0, :, :, 32])
        axs[0,1].imshow(img)
        axs[0,2].imshow(np.abs(gt[0, :, :, 32]-img), cmap='jet', vmin=0, vmax=1)

        thresholded_u1_gt = np.copy(gt[0, :, :, 32])
        thresholded_u2_gt = np.copy(gt[0, :, :, 32])
        thresholded_u1_gt[thresholded_u1_gt >= 0.7] = 1
        thresholded_u1_gt[thresholded_u1_gt < 0.7] = 0

        thresholded_u2_gt[thresholded_u2_gt >= 0.25] = 1
        thresholded_u2_gt[thresholded_u2_gt < 0.25] = 0

        axs[1,0].imshow(thresholded_u1_gt+thresholded_u2_gt)
        axs[1,1].imshow(thresholded_u1+thresholded_u2)
        axs[1,2].imshow(np.abs((thresholded_u1_gt+thresholded_u2_gt)-(thresholded_u1+thresholded_u2)), cmap='jet', vmin=0, vmax=1)

        plt.savefig('map_estimate_run1_round6.png')
    """