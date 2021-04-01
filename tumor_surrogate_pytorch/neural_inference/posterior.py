import GPUtil
import numpy as np
import os
import torch
import torchvision
import pickle
from sbi import utils
from sbi.analysis import eval_conditional_density
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT

from tumor_surrogate_pytorch.neural_inference.embedding_net import ConvNet
from tumor_surrogate_pytorch.neural_inference.simulator import Simulator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def print_gpu_utilisation():
    gpu = GPUtil.getGPUs()[1]
    print(f'{gpu.memoryUsed} MB allocated\n')

def plot_probabilities(ranges, gts, posterior, device, round):
    steps = 100
    for i, r in enumerate(ranges):
        x = np.linspace(r[0],r[1],steps)
        theta = np.tile(gts,(steps,1))
        theta[:,i] = x
        theta = torch.from_numpy(theta.astype(np.float32)).to(device)
        #plt.locator_params(axis="x", nbins=10)
        probs = np.exp(posterior.log_prob(theta).cpu().numpy())
        if i == 2:
            x = x * 20 + 1
        plt.axvline(x=gts[i], linestyle='dotted', color='red')
        plt.plot(x, probs)
        plt.savefig(f'tumor_surrogate_pytorch/neural_inference/plots/round_{round}_paramerter_{i}.png')
        plt.clf()
        plt.close()

def plot_probabilities_2(ranges, gts, posterior, device, round):
    steps = 100
    for i, r in enumerate(ranges):
        x = np.linspace(r[0]+1e-32,r[1]-1e-32,steps)
        probs = eval_conditional_density(posterior, torch.tensor(gts, device=device), torch.tensor(ranges, device=device), dim1=i, dim2=i, resolution=steps)
        #plt.locator_params(axis="x", nbins=10)
        if i == 2:
            x = x * 20 + 1
            plt.axvline(x=gts[i]*20+1, linestyle='dotted', color='red')
        else:
            plt.axvline(x=gts[i], linestyle='dotted', color='red')
        plt.plot(x, probs.cpu().numpy())
        plt.savefig(f'tumor_surrogate_pytorch/neural_inference/plots/round_{round}_paramerter_{i}.png')
        plt.clf()
        plt.close()

class NPE:
    def __init__(self, simulator, device, log_path, posterior_path=None):
        self.device = device
        self.prior = utils.BoxUniform(low=torch.tensor([0.0001, 0.0001, 0, 0.4, 0.4, 0.4, 0.6, 0.05], device=device),
                                      high=torch.tensor([0.0008, 0.03, 1, 0.6, 0.6, 0.6, 0.8, 0.6], device=device))
        self.posterior_path = posterior_path
        self.simulator = simulator
        self.posteriors = []
        self.writer = SummaryWriter(log_dir=log_path)

    def save_posterior(self, posterior, path):
        with open(path, "wb") as handle:
            pickle.dump(posterior, handle)

    def load_posterior(self, path):
        with open(path, "rb") as handle:
            posterior = pickle.load(handle)
        return posterior

    def bayesian_inference(self, num_samples, proposal):
        thetas = proposal.sample((num_samples,))
        tumor_density = torch.zeros(1,64,64,64)
        for i in range(thetas.shape[0]//32+1):
            curr_theta = thetas[i*32:(i+1)*32]
            tumor_density += self.simulator.predict_tumor_density(curr_theta)

        tumor_density /= num_samples
        return tumor_density

    def forward(self, x_ob, num_rounds, num_simulations, ranges, gts):
        print("Starting forward")
        neural_posterior = utils.posterior_nn(model='maf',
                                              embedding_net=ConvNet(device=self.device))
        inference = APT(prior=self.prior, device='gpu', density_estimator=neural_posterior)
        simulator, prior = prepare_for_sbi(self.simulator.predict_tumor_label_map, self.prior)
        proposal = prior
        if self.posterior_path is not None:
            proposal = self.load_posterior(self.posterior_path).set_default_x(x_ob)

        for i in range(num_rounds):
            print("Round: ", i)
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, simulation_batch_size=32)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(show_train_summary=True, training_batch_size=6,
                                                                                                num_atoms=6, discard_prior_samples=True, max_num_epochs=3)
            sample_with_mcmc = False  # True if i < 2 else False
            posterior = inference.build_posterior(density_estimator, sample_with_mcmc=sample_with_mcmc,
                                                  rejection_sampling_parameters={'max_sampling_batch_size': 50})
            self.posteriors.append(posterior)
            self.save_posterior(posterior, f'tumor_surrogate_pytorch/neural_inference/posteriors/round_{i}.pkl')

            proposal = posterior.set_default_x(x_ob)
            plot_probabilities_2(ranges, gts, proposal, device, i)
            tumor_density = self.bayesian_inference(1000, proposal)
            self.writer.add_image('bayesian plot', tumor_density[0:1,:,:,32], global_step=i)
            #map_estimate = proposal.map(num_init_samples=50, num_to_optimize=25, show_progress_bars=False)
            #for j in range(8):
            #    self.writer.add_scalar(f'parameter {j+1} map', map_estimate[j], global_step=i)
        return posterior


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    simulator = Simulator()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    npe = NPE(simulator=simulator, device=device, log_path='tumor_surrogate_pytorch/logs/run1')
    x_ob = np.load('tumor_surrogate_pytorch/neural_inference/x_obs_test.npz')
    x_ob = x_ob['x_025'] + x_ob['x_07']
    x_ob = x_ob.flatten()
    x_ob = torch.tensor(x_ob, device=device)
    posterior = npe.forward(x_ob=x_ob, num_rounds=10, num_simulations=5000,
                            ranges=[[0.0001,0.0008], [0.0001,0.03], [0,1], [0.4,0.6], [0.4,0.6], [0.4,0.6], [0.6,0.8], [0.05, 0.6]],
                            gts= [2.30e-04, 1.94e-02, 0.75, 4.37e-01, 5.36e-01, 4.91e-01, 0.7, 0.25])
