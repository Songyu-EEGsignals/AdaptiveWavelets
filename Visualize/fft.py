import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.fft as nf
# 导入所需库
def fft_tran(T,sr):
    complex_ary = nf.fft(sr)
    y_ = nf.ifft(complex_ary).real
    fft_freq = nf.fftfreq(y_.size, T[1] - T[0])
    fft_pow = np.abs(complex_ary)  # 复数的摸-Y轴
    return fft_freq, fft_pow