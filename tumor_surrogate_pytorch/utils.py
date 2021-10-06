import math
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def loss_function(u_sim, u_pred, csf):
    pred_loss = torch.mean(torch.abs(u_sim[u_sim >= 0.001] - u_pred[u_sim >= 0.001]))
    csf_loss = torch.mean(torch.abs(u_sim[csf >= 0.001] - u_pred[csf >= 0.001]))

    if math.isnan(pred_loss.item()):
        pred_loss = 0
    loss = pred_loss + csf_loss
    return loss

def weighted_loss(u_sim, u_pred, csf):
    pred_loss_tumor = torch.mean(torch.abs(u_sim[u_sim >= 0.001] - u_pred[u_sim >= 0.001]))
    pred_loss_healthy = torch.mean(torch.abs(u_sim[u_sim < 0.001] - u_pred[u_sim < 0.001]))

    if math.isnan(pred_loss_tumor.item()):
        pred_loss_tumor = torch.tensor(0)
    if math.isnan(pred_loss_healthy.item()):
        pred_loss_healthy = torch.tensor(0)

    pred_loss = 0.75 * pred_loss_tumor + 0.25 * pred_loss_healthy
    csf_loss = torch.mean(torch.abs(u_sim[csf >= 0.001] - u_pred[csf >= 0.001]))

    if math.isnan(pred_loss.item()):
        pred_loss = 0

    loss = pred_loss + csf_loss
    return loss


def compute_dice_score(u_pred, u_sim, threshold):
    tp = torch.sum((u_pred > threshold) * (u_sim > threshold)).float()
    tpfp = torch.sum(u_pred > threshold).float()
    tpfn = torch.sum(u_sim > threshold).float()
    if tpfn + tpfp == 0:
        return None
    return torch.mean(2 * tp / (tpfn + tpfp))

def mean_absolute_error_helper(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_error(ground_truth, output, input):

    input[:, 0][input[:, 0] >= 0.35] = 1
    input[:, 0][input[:, 0] < 0.35] = 0
    input[:, 1][input[:, 1] > 0.3] = 1
    input[:, 1][input[:, 1] <= 0.3] = 0
    input[:, 1][input[:, 0] >= 0.35] = 0

    ground_truth = np.array(ground_truth[:, 0].detach().cpu())
    output = np.array(output[:, 0].detach().cpu())
    input = input.cpu()
    wm = np.ma.masked_where(np.logical_and(input[:, 0] > 0.0001, ground_truth > 0.0001), input[:, 0])
    mae_wm = mean_absolute_error_helper(output[wm.mask].ravel(), ground_truth[wm.mask].ravel())

    gm = np.ma.masked_where(np.logical_and(input[:, 1] > 0.0001, ground_truth > 0.0001), input[:, 1])
    mae_gm = mean_absolute_error_helper(output[gm.mask].ravel(), ground_truth[gm.mask].ravel())

    csf = np.ma.masked_where(np.logical_and(input[:, 2] > 0.0001, output > 0.0001), input[:, 2])
    mae_csf = mean_absolute_error_helper(output[csf.mask].ravel(), ground_truth[csf.mask].ravel())

    return mae_wm, mae_gm, mae_csf



def create_hists(model, val_loader, device, save_path):
    model.eval()
    mae_wm = []
    mae_gm = []
    mae_csf = []
    dice_score_02 = []
    dice_score_04 = []
    dice_score_08 = []
    losses = []
    data = []

    with torch.no_grad():
        print("Dataloader lenght: ", len(val_loader))
        for i, (input_batch, parameters, ground_truth_batch) in enumerate(val_loader):
            print(f'iteration {i} of {len(val_loader)}')
            input_batch, parameters, ground_truth_batch = input_batch.to(device), parameters.to(device), ground_truth_batch.to(device)
            # compute output
            output_batch, attmaps = model(input_batch, parameters)
            # measure mae, dice score and record loss
            loss = loss_function(u_sim=ground_truth_batch, u_pred=output_batch, csf=input_batch[:,2:3])
            losses.append(loss.item())

            for output, ground_truth, input in zip(output_batch, ground_truth_batch, input_batch):
                output = output[None]
                ground_truth = ground_truth[None]
                input = input[None]
                # visualize input and output - very slow, only activate when needed
                # fig, axs = plt.subplots(1, 5, sharey=False)
                # wm = input[0,0, :, :, 32]
                # gm = input[0,1, :, :, 32]
                # csf = input[0,2, :, :, 32]
                # axs[0].imshow(wm.cpu())
                # axs[1].imshow(gm.cpu())
                # axs[2].imshow(csf.cpu())
                # axs[3].imshow(ground_truth[0,0, :, :, 32].cpu())
                # axs[4].imshow(output[0,0,:,:,32].cpu())
                # plt.show()

                #visualize attention map
                #plt.imshow(attmaps[0][00, :, :, 32].cpu().numpy(), cmap='jet')
                #plt.show()

                dice_02 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.2)
                dice_04 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                dice_08 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.8)

                input = input.cpu()
                #mae_wm_value0, mae_gm_value0, mae_csf_value0 = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)

                if mae_wm_value is not None:
                    mae_wm.append(mae_wm_value)

                if mae_gm_value is not None:
                    mae_gm.append(mae_gm_value)

                if mae_csf_value is not None:
                    mae_csf.append(mae_csf_value)

                if dice_02 is not None:
                    dice_score_02.append(dice_02.item())
                if dice_04 is not None:
                    dice_score_04.append(dice_04.item())
                if dice_08 is not None:
                    dice_score_08.append(dice_08.item())

                data.append([dice_to_num(dice_02),
                             dice_to_num(dice_04),
                             dice_to_num(dice_08),
                             mae_to_num(mae_wm_value),
                             mae_to_num(mae_gm_value),
                             mae_to_num(mae_csf_value)])

    print(sum(losses)/len(losses))

    Path(save_path).mkdir(parents=True, exist_ok=True)

    #delete zeros

    print("Dice 20: ", np.array(dice_score_02).mean())
    print("Dice 20 no zeros: ", np.array(dice_score_02)[np.array(dice_score_02) > 0.001].mean())
    print("Dice 40: ",np.array(dice_score_04).mean())
    print("Dice 40 no zeros: ", np.array(dice_score_04)[np.array(dice_score_04) > 0.001].mean())
    print("Dice 80: ",np.array(dice_score_08).mean())
    print("Dice 80 no zeros: ", np.array(dice_score_08)[np.array(dice_score_08) > 0.001].mean())
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(dice_score_02, bins=50)
    axs[1].hist(dice_score_04, bins=50)
    axs[2].hist(dice_score_08, bins=50)
    plt.savefig(save_path + 'dice.png')

    print("MAE WM: ",np.array(mae_wm).mean())
    print("MAE GM: ",np.array(mae_gm).mean())
    print("MAE CSF: ",np.array(mae_csf).mean())
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    axs[0].hist(mae_wm, bins=50)
    axs[1].hist(mae_gm, bins=50)
    axs[2].hist(mae_csf, bins=50)
    plt.savefig(save_path + 'mae.png')

    data.append(['Dice20', np.array(dice_score_02).mean()])
    data.append(['Dice20 no zeros', np.array(dice_score_02)[np.array(dice_score_02) > 0.001].mean()])
    data.append(['Dice40', np.array(dice_score_04).mean()])
    data.append(['Dice40 no zeros', np.array(dice_score_04)[np.array(dice_score_04) > 0.001].mean()])
    data.append(['Dice80', np.array(dice_score_08).mean()])
    data.append(['Dice80 no zeros', np.array(dice_score_08)[np.array(dice_score_08) > 0.001].mean()])
    data.append(['MAE WM: ', np.array(mae_wm).mean()])
    data.append(['MAE GM: ', np.array(mae_gm).mean()])
    data.append(['MAE CSF: ', np.array(mae_csf).mean()])
    df = pd.DataFrame(data, columns=['Dice20', 'Dice40', 'Dice80', 'MAE WM', 'MAE GM', 'MAE CSF'])
    df.to_csv(save_path + 'stats.csv')
def mae_to_num(mae):
    if mae is None:
        return 'None'
    else:
        return mae

def dice_to_num(dice):
    if dice is None:
        return 'None'
    else:
        return dice.item()