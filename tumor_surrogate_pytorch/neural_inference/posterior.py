import sbi
import torch
from sbi import utils
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT

from tumor_surrogate_pytorch.neural_inference.simulator import Simulator
import numpy as np


class NPE:
    def __init__(self, simulator):
        self.prior = utils.BoxUniform(low=torch.zeros(6),
                                 high=torch.ones(6))
        self.simulator = simulator
        self.posteriors = []

    def forward(self, x_ob, num_rounds, num_simulations):
        inference = APT(prior=self.prior) #TODO set correct density estimator or self-defined model
        simulator, prior = prepare_for_sbi(self.simulator, self.prior)
        proposal = prior
        posterior = None
        for _ in range(num_rounds):
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(x_ob)
        return posterior

if __name__ == '__main__':
    simulator = Simulator()
    npe = NPE(simulator = simulator)
    x_ob = np.load('tumor_surrogate_pytorch/neural_inference/x_obs_test.npz')
    x_ob = x_ob['x_025'] + x_ob['x_07']
    posterior = npe.forward(x_ob=x_ob, num_rounds=10, num_simulations=500)