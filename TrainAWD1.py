from matplotlib import pyplot as plt, gridspec

from AWD_model.utils.misc import get_wavefun
from TrainPremodel0 import batch_size, num_epochs,  premodel_name, premodel_path
import numpy as np
import random, os
import torch
import pickle as pkl
from AWD_model.EEGNet import EEGNet

from AWD_model.DWT1d import DWT1d
from AWD_model.DWT2d import DWT2d
from AWD_model.losses import get_loss_f
from AWD_model.utils.train import Trainer
from AWD_model.utils.evaluate import Validator



device = 'cuda' if torch.cuda.is_available() else 'cpu'
opj = os.path.join
#----------------------------
seed = 1
wave_rows = ["db3","db4","db5","db6","db7","sym3","sym4","sym5","sym6","sym7"]
wave_column = None
J = 3
mode = 'periodization'
init_factor = 1
noise_factor = 0
const_factor = 0
batch_size = batch_size
lr = 0.001
num_epochs = 50
attr_methods = 'Saliency'

lamL1wave = 0.1
lamL1attr = 0.0

# lamL1attrs = np.round([0] + list(np.geomspace(0.01, 5, 20)), 5)
target = -1
dirname = 'db4_saliency_warmstart_mode=per_cnn_seed=1'
warm_start = True
data_path = "./data_path"
model_path = "./result_premodel"
wt_type = 'DWT2d'

lamlSum = 1
lamhSum = 1
lamL2norm = 1
lamCMF = 1
lamConv = 1
out_dir = "./result"
pid = ''.join(["%s" % random.randint(0, 9) for num in range(0, 10)])

import pickle

with open(r'./data_path/train_sub01.pkl', 'rb') as f:
    train_loader = pickle.load(f)
with open(r'./data_path/test_sub01.pkl', 'rb') as f:
    test_loader = pickle.load(f)


#----------------------------
#class



def load_pretrained_model(premodel_path, premodel_name, device=device, EEGNet=EEGNet):
    """load pretrained model for interpretation
    """
    eegnet = EEGNet().to(device)
    eegnet.load_state_dict(torch.load(opj(premodel_path, premodel_name)))
    eegnet = eegnet.eval()
    # freeze layers
    for param in eegnet.parameters():
        param.requires_grad = False

    return eegnet

def fun_warm_start(out_dir, wave_row, wave_column, mode, J, init_factor, noise_factor, lamL1attr, lamL1wave, wave=None):
    '''load results and initialize model
    '''
    print('\twarm starting...')
    # fnames = sorted(os.listdir(out_dir))
    fnames = []
    _lamL1attr = []
    _lamL1wave = []
    models = []
    if len(fnames) == 0:
        if wt_type == 'DWT1d':
            _model = DWT1d(wave=wave, mode=mode, J=J, init_factor=init_factor, noise_factor=noise_factor).to(
                device)
        elif wt_type == 'DWT2d':
            _model = DWT2d(wave_row=wave_row, wave_column=wave_column, mode=mode, J=J, init_factor=init_factor, noise_factor=noise_factor).to(
                device)
    else:
        for fname in fnames:
            if fname[-3:] == 'pkl':
                result = pkl.load(open(opj(out_dir, fname), 'rb'))
                _lamL1attr.append(result['lamL1attr'])
                _lamL1wave.append(result['lamL1wave'])
            if fname[-3:] == 'pth':
                if wt_type == 'DWT1d':
                    m = DWT1d(wave=wave, mode=mode, J=J, init_factor=init_factor,
                              noise_factor=noise_factor).to(device)
                elif wt_type == 'DWT2d':
                    m = DWT2d(wave_row=wave_row, wave_column=wave_column, mode=mode, J=J, init_factor=init_factor,
                              noise_factor=noise_factor).to(device)
                m.load_state_dict(torch.load(opj(out_dir, fname)))
                models.append(m)
        _lamL1attr = np.array(_lamL1attr)
        _lamL1wave = np.array(_lamL1wave)
        if lamL1attr == 0:
            _lamL1wave_max = np.max(_lamL1wave[_lamL1attr == 0])
            idx = np.argwhere((_lamL1attr == 0) & (_lamL1wave == _lamL1wave_max)).item()
        else:
            _lamL1attr_max = np.max(_lamL1attr[_lamL1wave == lamL1wave])
            idx = np.argwhere((_lamL1attr == _lamL1attr_max) & (_lamL1wave == lamL1wave)).item()
        _model = models[idx]
        print('initialized at the model with lamL1wave={:.5f} and lamL1attr={:.5f}'.format(_lamL1wave[idx],
                                                                                           _lamL1attr[idx]))
    return _model

class s:
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                if not attr.startswith('_')}




def run(wave_row):
    waveimglist = []
    model = load_pretrained_model(premodel_path=premodel_path, premodel_name=premodel_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if warm_start is None:
        wt = DWT2d(wave_row=wave_row, wave_column=wave_column, mode=mode, J=J,
                   init_factor=init_factor,
                   noise_factor=noise_factor,
                   const_factor=const_factor).to(device)
        wt.train()
    else:
        wt = fun_warm_start(out_dir=out_dir, wave_row=wave_row, wave_column=wave_column, mode=mode, J=J,
                            init_factor=init_factor,
                            noise_factor=noise_factor, lamL1attr=lamL1attr, lamL1wave=lamL1wave).to(device)
        wt.train()

    # check if we have multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        wt = torch.nn.DataParallel(wt)

    params = list(wt.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_f = get_loss_f(lamlSum=lamlSum, lamhSum=lamhSum, lamL2norm=lamL2norm, lamCMF=lamCMF, lamConv=lamConv,
                        lamL1wave=lamL1wave, lamL1attr=lamL1attr)
    trainer = Trainer(model, wt, optimizer, loss_f, target=target,
                      use_residuals=True, attr_methods=attr_methods, device=device, n_print=5)
    # run
    waveimglist = trainer(train_loader, test_loader, epochs=num_epochs)
    # calculate losses
    print('calculating losses and metric...')
    validator = Validator(model, test_loader)
    rec_loss, lsum_loss, hsum_loss, L2norm_loss, CMF_loss, conv_loss, L1wave_loss, L1saliency_loss, L1inputxgrad_loss = validator(
        wt, target=target)
    s.train_losses = trainer.train_losses
    s.rec_loss = rec_loss
    s.lsum_loss = lsum_loss
    s.hsum_loss = hsum_loss
    s.L2norm_loss = L2norm_loss
    s.CMF_loss = CMF_loss
    s.conv_loss = conv_loss
    s.L1wave_loss = L1wave_loss
    s.L1saliency_loss = L1saliency_loss
    s.L1inputxgrad_loss = L1inputxgrad_loss
    s.net = wt

    # save
    pkl_name = 'wave=' + str(wave_row) + str(wave_column) + '_lamL1wave=' + str(lamL1wave) + '_lamL1attr=' + str(
        lamL1attr) \
               + '_seed=' + str(seed) + '_pid=' + pid
    results = {'lamL1attr': lamL1attr, 'lamlSum': lamlSum, 'lamhSum': lamhSum, 'lamL2norm': lamL2norm,
               'lamCMF': lamCMF, 'lamConv': lamConv, 'lamL1wave': lamL1wave, 'lamL1attr': lamL1attr, **s._dict(s)}
    pkl.dump(results, open(opj(out_dir, pkl_name + '.pkl'), 'wb'))
    if torch.cuda.device_count() > 1:
        torch.save(wt.module.state_dict(), opj(out_dir, pkl_name + '.pth'))
    else:
        torch.save(wt.state_dict(), opj(out_dir, pkl_name + '.pth'))

    R = len(waveimglist)
    print(R)

    psi_list = []
    phi_list = []
    for r in waveimglist[0], waveimglist[-1]:
        phi, psi, x = get_wavefun(r)
        phi_list.append(phi)
        psi_list.append(psi)

    plt.figure(dpi=200)
    plt.plot(x, phi_list[0], color='blue', linewidth=2)
    plt.plot(x, phi_list[1], color='red', linewidth=2)
    plt.legend(['Initial symN wavelets', 'Wavelets after training'], loc='upper right')
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False)
    plt.ylabel(str(lamL1attr))
    plt.title(str(num_epochs))
    plt.savefig('figures/' + str(wave_row) + str(lamL1attr) + "-phi-" + str(num_epochs) + '.pdf',
                bbox_inches='tight')
    plt.close()
    # plt.show()

    plt.figure(dpi=200)
    plt.plot(x, psi_list[0], color='blue', linewidth=2)
    plt.plot(x, psi_list[1], color='red', linewidth=2)
    plt.legend(['Initial ' + str(wave_row) + ' wavelets', 'Wavelets after training'], loc='upper right')
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False)
    plt.ylabel(str(lamL1attr))
    plt.title(str(num_epochs))
    plt.savefig('figures/' + str(wave_row)  + "-psi-" + str(num_epochs) + "-loss-0720-" + str(
        rec_loss) + '.pdf', bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    for wave_row in wave_rows:
        run(wave_row=wave_row)


