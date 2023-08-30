import torch
import torch.nn as nn
import torch.optim
from AWD_model.AbstractWT import AbstractWT
from AWD_model.utils import lowlevel
from AWD_model.utils.misc import init_filter, low_to_high

class DWT2d(AbstractWT):
    '''Class of 2d wavelet transform 
    Params
    ------
    J: int
        number of levels of decomposition
    wave: str
         which wavelet to use.
         can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
    mode: str
        'zero', 'symmetric', 'reflect' or 'periodization'. The padding scheme
    '''

    def __init__(self, wave_row='db5', wave_column=None, mode='zero', J=3, init_factor=1, noise_factor=0, const_factor=0, device='cpu'):
        super().__init__()
        row, _ = lowlevel.load_wavelet(wave_row)
        # row = torch.rand(1, 1, 14)

        # initialize
        row = init_filter(row, init_factor, noise_factor, const_factor)

        # parameterize
        self.row = nn.Parameter(row, requires_grad=True)
        self.row = self.row.to(device)

        self.wave_column = wave_column


        if self.wave_column is not None:
            column, _ = lowlevel.load_wavelet(wave_column)
            column = init_filter(column, init_factor, noise_factor, const_factor)
            self.column = nn.Parameter(column, requires_grad=True)
            self.column = self.column.to(device)

        self.J = J
        self.mode = mode
        self.wt_type = 'DWT2d'
        self.device = device

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = ()
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        row1 = low_to_high(self.row)
        row = self.row.reshape((1, 1, 1, -1))
        row1 = row1.reshape((1, 1, 1, -1))

        # if self.wave_column is not None:
        #     column1 = low_to_high(self.column)
        #
        #     column = self.column.reshape((1, 1, -1, 1))
        #     column1 = column1.reshape((1, 1, -1, 1))

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            # ll, high = lowlevel.AFB2D.forward(
            #     ll, column, column1, row, row1, mode)
            ll, high = lowlevel.AFB2D.forward(
                ll, row, row1, mode)

            yh += (high,)


        return (ll,) + yh

    def inverse(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        coeffs = list(coeffs)
        yl = coeffs.pop(0)
        yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        row1 = low_to_high(self.row)

        g0_row = self.row.reshape((1, 1, 1, -1))
        g1_row = row1.reshape((1, 1, 1, -1))

        # if self.wave_column is not None:
        #     column1 = low_to_high(self.column)
        #     g0_col = self.column.reshape((1, 1, -1, 1))
        #     g1_col = column1.reshape((1, 1, -1, 1))

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            ll = lowlevel.SFB2D.forward(
                ll, h,  g0_row, g1_row, mode)
        return ll

if __name__=="__main__":
    A = DWT2d()
