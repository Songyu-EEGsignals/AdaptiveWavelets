import matplotlib.pyplot as plt
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
opj = os.path.join
import pickle as pkl
import pandas as pd
from AWD_model.DWT2d import DWT2d
from AWD_model.utils.misc import get_wavefun
from matplotlib import gridspec
import TrainAWD1

#----------------------------
data_name = "test1"
#----------------------------

if __name__ == '__main__':
    results = []
    models = []
    # load results
    out_dir = "./result"
    fnames = sorted(os.listdir(out_dir))
    results_list = []
    models_list = []
    for fname in fnames:
        if fname[-3:] == 'pkl':
            results_list.append(pkl.load(open(opj(out_dir, fname), 'rb')))
        if fname[-3:] == 'pth':
            wt = DWT2d(wave_row=TrainAWD1.wave_row, wave_column=TrainAWD1.wave_column, mode='zero', J=TrainAWD1.J, init_factor=1, noise_factor=0.0).to(device)
            wt.load_state_dict(torch.load(opj(out_dir, fname)))
            models_list.append(wt)
    pd_file = pd.DataFrame(results_list)
    pd_file.to_excel('./result_excel/' + data_name + '.xlsx', index=False)
    results.append(pd_file)
    models.append(models_list)
    res = results[0]
    mos = models[0]
    lamL1wave = np.array(res['lamL1wave'])
    lamL1attr = np.array(res['lamL1attr'])
    lamL1wave_grid = np.unique(lamL1wave)
    lamL1attr_grid = np.unique(lamL1attr)
    index2o = {}
    index2t = {}
    num = 0
    for i, _ in enumerate(lamL1wave_grid):
        for j, _ in enumerate(lamL1attr_grid):
            loc = (lamL1wave == lamL1wave_grid[i]) & (lamL1attr == lamL1attr_grid[j])
            if loc.sum() == 1:
                loc = np.argwhere(loc).flatten()[0]
                index2o[(i, j)] = loc
                index2t[num] = (i, j)
            num += 1
    R = len(lamL1wave_grid)
    C = len(lamL1attr_grid)
    psi_list = []
    wt_list = []
    for r in range(R):
        for c in range(C):
            print(index2o)
            print(mos)
            wt = mos[index2o[(r, c)]]
            print("-------------")
            print(wt)
            wt_list.append(wt)
            phi, psi, x = get_wavefun(wt)
            psi_list.append(psi)
    plt.figure(figsize=(C + 1, R + 1), dpi=200)
    gs = gridspec.GridSpec(R, C,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (R + 1), bottom=0.5 / (R + 1),
                           left=0.5 / (C + 1), right=1 - 0.5 / (C + 1))
    i = 0
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(gs[r, c])
            ax.plot(x, psi_list[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False)
            if c == 0:
                plt.ylabel(str(lamL1wave_grid[r]))
            if r == 0:
                plt.title(str(lamL1attr_grid[c]))
            i += 1
    plt.savefig('figures/' + data_name + '.pdf', bbox_inches='tight')
    plt.show()


