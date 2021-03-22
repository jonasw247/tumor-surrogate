import os

import sbi
import torch
import numpy as np
from sbi import utils
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT

from tumor_surrogate_pytorch.neural_inference.density_estimator import build_tumorde
from tumor_surrogate_pytorch.neural_inference.simulator import Simulator
import GPUtil


def print_gpu_utilisation():
    gpu = GPUtil.getGPUs()[1]
    print(f'{gpu.memoryUsed} MB allocated\n')

class NPE:
    def __init__(self, simulator):
        self.prior = utils.BoxUniform(low=torch.zeros(8),
                                 high=torch.ones(8))
        self.simulator = simulator
        self.posteriors = []

    def forward(self, x_ob, num_rounds, num_simulations):
        print("Starting forward")
        inference = APT(prior=self.prior, device='gpu', density_estimator=build_tumorde)
        simulator, prior = prepare_for_sbi(self.simulator, self.prior)
        proposal = prior
        for i in range(num_rounds):
            print("Round: ", i)
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations)
            theta = theta.to(torch.device('cuda'))
            x = x.to(torch.device('cuda'))
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(show_train_summary=True)
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(x_ob)
        return posterior

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    simulator = Simulator()
    npe = NPE(simulator = simulator.predict_tumor_density)
    x_ob = np.load('tumor_surrogate_pytorch/neural_inference/x_obs_test.npz')
    x_ob = x_ob['x_025'] + x_ob['x_07']
    x_ob = x_ob.flatten()
    x_ob = torch.tensor(x_ob).to(torch.device('cuda'))
    posterior = npe.forward(x_ob=x_ob, num_rounds=3, num_simulations=10)
