import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_probabilities(ranges, gts):
    steps = 100
    for i, r in enumerate(ranges):
        x = np.linspace(r[0],r[1],steps)
        theta = np.tile(gts,(steps,1))
        theta[:,i] = x
        #plt.locator_params(axis="x", nbins=10)
        probs = np.exp(torch.sin(torch.from_numpy(theta)))
        plt.axvline(x=gts[i], linestyle='dotted', color='red')
        plt.plot(x, probs[:,0])
        plt.show()


if __name__ == '__main__':
    #xmin, xmax = -1, 5
    #x = np.linspace(-15, 15, 100)  # 100 linearly spaced numbers
    #y = np.sin(x) / x  # computing the values of sin(x)/x
    #plt.locator_params(axis="x", nbins=10)
    #plt.axvline(x=2.20589566, linestyle='dotted', color='red')
    #plt.plot(x, y)
    #plt.show()
    #plot_probabilities(ranges=[[0.0001,0.0008], [0.0001,0.03], [0,1], [0.4,0.6], [0.4,0.6], [0.4,0.6], [0.6,0.8], [0.05, 0.6]],
    #                        gts= [2.30e-04, 1.94e-02, 1.60e+01, 4.37e-01, 5.36e-01, 4.91e-01, 0.7, 0.25])

        for param in range(8):
            fig, axs = plt.subplots(1,7, sharey=True, tight_layout=True, figsize=(30,5))
            for round in range(7):
                axs[round].axis('off')
                axs[round].get_xaxis().set_visible(False)
                axs[round].get_yaxis().set_visible(False)
                img = plt.imread(f'tumor_surrogate_pytorch/neural_inference/output/run1/plots/round_{round}_paramerter_{param}.png')
                axs[round].imshow(img)
            #plt.show()
            plt.savefig(f'development_param_{param}.png')

