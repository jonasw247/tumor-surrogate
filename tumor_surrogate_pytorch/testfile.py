import pandas as pd
import numpy as np
import random
import numpy
from matplotlib import pyplot as plt
import json

if __name__ == '__main__':

    data = pd.read_csv('test_output/whole_dataset_orig_loss_best_other_mae_method2/stats.csv')
    mae_wm = data['MAE WM'][:10000]
    num_nones = sum(np.isnan(mae_wm))
    print(f"MAE WM: {mae_wm.size} elements,  Nans: {num_nones}")

    mae_gm = data['MAE GM'][:10000]
    num_nones = sum(np.isnan(mae_gm))
    print(f"MAE GM: {mae_gm.size} elements,  Nans: {num_nones}")

    mae_csf = data['MAE CSF'][:10000]
    num_nones = sum(np.isnan(mae_csf))
    print(f"MAE CSF: {mae_csf.size} elements,  Nans: {num_nones}")

    dice20 = data['Dice20'][:10000]
    num_nones = np.sum(dice20.to_numpy() == 'None')
    print(f"Dice20: {dice20.size} elements,  Nans: {num_nones}")

    dice40 = data['Dice40'][:10000]
    num_nones = np.sum(dice40.to_numpy() == 'None')
    print(f"Dice40: {dice40.size} elements,  Nans: {num_nones}")

    dice80 = data['Dice80'][:10000]
    num_nones = np.sum(dice80.to_numpy() == 'None')
    print(f"Dice80: {dice80.size} elements,  Nans: {num_nones}")

    print("MAE wm: ", np.nanmean(np.array(mae_wm)))
    print("MAE gm: ", np.nanmean(np.array(mae_gm)))
    print("MAE csf: ", np.nanmean(np.array(mae_csf)))

    dice20 = np.array(np.array(dice20)[np.array(dice20) != 'None'], dtype=np.float)
    dice40 = np.array(np.array(dice40)[np.array(dice40) != 'None'], dtype=np.float)
    dice80 = np.array(np.array(dice80)[np.array(dice80) != 'None'], dtype=np.float)
    print("Dice 20: ", dice20.mean())
    print("Dice 20 no zeros: ", np.array(dice20)[np.array(dice20) > 0.001].mean())
    print("Dice 40: ", dice20.mean())
    print("Dice 40 no zeros: ", np.array(dice40)[np.array(dice40) > 0.001].mean())
    print("Dice 80: ", dice80.mean())
    print("Dice 80 no zeros: ", np.array(dice80)[np.array(dice80) > 0.001].mean())



    # original model data
    set = 'test'
    with open(f'orig_output/{set}set/dice20_{set}set.json', 'r') as f:
        dice20_orig = json.load(f)

    with open(f'orig_output/{set}set/dice40_{set}set.json', 'r') as f:
        dice40_orig = json.load(f)

    with open(f'orig_output/{set}set/dice80_{set}set.json', 'r') as f:
        dice80_orig = json.load(f)

    with open(f'orig_output/{set}set/mae_wm_{set}set.json', 'r') as f:
        mae_wm_orig = json.load(f)

    with open(f'orig_output/{set}set/mae_gm_{set}set.json', 'r') as f:
        mae_gm_orig = json.load(f)

    with open(f'orig_output/{set}set/mae_csf_{set}set.json', 'r') as f:
        mae_csf_orig = json.load(f)

    # plot joined Histogram
    #mae
    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True, figsize=(15,5))

    bins = numpy.linspace(0, 0.2, 50)

    axs[0].hist(list(mae_wm), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[0].hist(list(mae_wm_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[0].legend(loc='upper right', title="WM")

    bins = numpy.linspace(0, 0.2, 50)

    axs[1].hist(list(mae_gm), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[1].hist(list(mae_gm_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[1].legend(loc='upper right', title="GM")

    bins = numpy.linspace(0, 0.1, 50)

    axs[2].hist(list(mae_csf), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[2].hist(list(mae_csf_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[2].legend(loc='upper right', title="CSF")

    plt.savefig('mae_hist.png')

    #dice
    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True, figsize=(15,5))

    bins = numpy.linspace(0, 1, 50)

    axs[0].hist(list(dice20), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[0].hist(list(dice20_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[0].legend(loc='upper left', title="DICE 0.2")

    bins = numpy.linspace(0, 1, 50)

    axs[1].hist(list(dice40), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[1].hist(list(dice40_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[1].legend(loc='upper left', title="DICE 0.4")

    bins = numpy.linspace(0, 1, 50)

    axs[2].hist(list(dice80), bins, alpha=0.5, label='ATT', color='r', ec='black')
    axs[2].hist(list(dice80_orig), bins, alpha=0.5, label='Ezhov et al.', color='cornflowerblue', ec='black')
    axs[2].legend(loc='upper left', title="DICE 0.8")

    plt.savefig('dice_hist.png')