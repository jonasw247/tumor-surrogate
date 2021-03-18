import sbi
import torch
from sbi import utils
from sbi.inference import simulate_for_sbi, prepare_for_sbi, APT


def simulate_parameters_from_uniform(parameters):
    parameters[0] = parameters[0] * 0.0007 + 0.0001
    parameters[1] = parameters[1] * 0.0299 + 0.0001
    parameters[2] = int(parameters[2] * 20 + 1)
    parameters[6] = parameters[6] * 0.2 + 0.6
    parameters[7] = parameters[7] * 0.55 + 0.05
    return parameters


class NPE:
    def __init__(self, simulator):
        self.prior = utils.BoxUniform(low=torch.zeros(8),
                                 high=torch.ones(8))
        self.simulator = simulator
        self.posteriors = []

    def forward(self, x_ob, num_rounds, num_simulations):
        inference = APT(prior=self.prior)
        simulator, prior = prepare_for_sbi(self.simulator, self.prior)
        proposal = prior
        for _ in range(num_rounds):
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations)
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(x_ob)

if __name__ == '__main__':
    print("hier")
