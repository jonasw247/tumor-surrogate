import GPUtil
import numpy as np
import os
import torch
from sbi import utils
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT

from tumor_surrogate_pytorch.neural_inference.embedding_net import ConvNet
from tumor_surrogate_pytorch.neural_inference.simulator import Simulator

def print_gpu_utilisation():
    gpu = GPUtil.getGPUs()[1]
    print(f'{gpu.memoryUsed} MB allocated\n')


class NPE:
    def __init__(self, simulator, device):
        self.device = device
        self.prior = utils.BoxUniform(low=torch.tensor([0.0001, 0.0001, 0, 0.4, 0.4, 0.4, 0.6, 0.05], device=device),
                                      high=torch.tensor([0.0008, 0.03, 1, 0.6, 0.6, 0.6, 0.8, 0.6], device=device))
        self.simulator = simulator
        self.posteriors = []

    def forward(self, x_ob, num_rounds, num_simulations):
        print("Starting forward")
        neural_posterior = utils.posterior_nn(model='maf',
                                              embedding_net=ConvNet(device=self.device))
        inference = APT(prior=self.prior, device='gpu', density_estimator=neural_posterior)
        simulator, prior = prepare_for_sbi(self.simulator, self.prior)
        proposal = prior
        for i in range(num_rounds):
            print("Round: ", i)
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations, simulation_batch_size=32)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(show_train_summary=True, training_batch_size=6,
                                                                                                num_atoms=6, discard_prior_samples=True)
            sample_with_mcmc = False  # True if i < 2 else False
            posterior = inference.build_posterior(density_estimator, sample_with_mcmc=sample_with_mcmc,
                                                  rejection_sampling_parameters={'max_sampling_batch_size': 50})
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(x_ob)
            map = proposal.map(num_init_samples=50, num_to_optimize=25, show_progress_bars=False)
            print(map)
        return posterior


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    simulator = Simulator()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    npe = NPE(simulator=simulator.predict_tumor_density, device=device)
    x_ob = np.load('tumor_surrogate_pytorch/neural_inference/x_obs_test.npz')
    x_ob = x_ob['x_025'] + x_ob['x_07']
    x_ob = x_ob.flatten()
    x_ob = torch.tensor(x_ob, device=device)
    posterior = npe.forward(x_ob=x_ob, num_rounds=10, num_simulations=5000)
