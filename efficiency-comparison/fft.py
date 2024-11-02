# @T.C.: Performing 2D convolution via Fast Fourier Transform.

import torch
import torch.fft as fft

def FFT(fourier_coef1, fourier_coef2, return_real = False):
    '''
    Performing 2D convolution via Fast Fourier Transform.
    i.e., res^*_{s,t}=\sum_{s_1+s_2=s,t_1+t_2=t}fourier_coef1^*_{1,s_1,t_1}*fourier_coef2^*_{2,s_2,t_2}.
    '''
    
    # Step 0: preparation
    in_shape1, in_shape2 = fourier_coef1.shape[0], fourier_coef2.shape[0]
    out_shape = in_shape1 + in_shape2 -1
    in1 = torch.zeros((out_shape, out_shape), dtype = fourier_coef1.dtype, device = fourier_coef1.device)
    in2 = torch.zeros((out_shape, out_shape), dtype = fourier_coef2.dtype, device = fourier_coef2.device)
    in1[:in_shape1, :in_shape1] = fourier_coef1
    in2[:in_shape2, :in_shape2] = fourier_coef2
    
    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fourier_coef1_freq, fourier_coef2_freq = fft.fft2(in1), fft.fft2(in2)
    
    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fourier_coef1_freq * fourier_coef2_freq
    
    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)
    
    return res

def FFT_channel(fourier_coef1, fourier_coef2, return_real = False):
    '''
    Performing 2D convolution via Fast Fourier Transform (with more than one channels).
    i.e., res^*_{s,t}=\sum_{s_1+s_2=s,t_1+t_2=t}fourier_coef1^*_{1,s_1,t_1}*fourier_coef2^*_{2,s_2,t_2}.
    '''
    
    # Step 0: preparation
    C = fourier_coef1.shape[0]
    in_shape1, in_shape2 = fourier_coef1.shape[1], fourier_coef2.shape[1]
    out_shape = in_shape1 + in_shape2 -1
    in1 = torch.zeros((C, out_shape, out_shape), dtype = fourier_coef1.dtype, device = fourier_coef1.device)
    in2 = torch.zeros((C, out_shape, out_shape), dtype = fourier_coef2.dtype, device = fourier_coef2.device)
    in1[:, :in_shape1, :in_shape1] = fourier_coef1
    in2[:, :in_shape2, :in_shape2] = fourier_coef2
    
    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fourier_coef1_freq, fourier_coef2_freq = fft.fft2(in1), fft.fft2(in2)
    
    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fourier_coef1_freq * fourier_coef2_freq
    
    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)
    
    return res

def FFT_batch_channel(fourier_coef1, fourier_coef2, return_real = False):
    '''
    Performing 2D convolution via Fast Fourier Transform (with more than one batches and channels).
    i.e., res^*_{s,t}=\sum_{s_1+s_2=s,t_1+t_2=t}fourier_coef1^*_{1,s_1,t_1}*fourier_coef2^*_{2,s_2,t_2}.
    '''
    
    # Step 0: preparation
    B, C = fourier_coef1.shape[0], fourier_coef1.shape[1]
    in_shape1, in_shape2 = fourier_coef1.shape[2], fourier_coef2.shape[2]
    out_shape = in_shape1 + in_shape2 -1
    in1 = torch.zeros((B, C, out_shape, out_shape), dtype = fourier_coef1.dtype, device = fourier_coef1.device)
    in2 = torch.zeros((B, C, out_shape, out_shape), dtype = fourier_coef2.dtype, device = fourier_coef2.device)
    in1[:, :, :in_shape1, :in_shape1] = fourier_coef1
    in2[:, :, :in_shape2, :in_shape2] = fourier_coef2
    
    # Step 1: 2D Discrete Fourier Transform: transform into the frequency domain
    fourier_coef1_freq, fourier_coef2_freq = fft.fft2(in1), fft.fft2(in2)
    
    # Step 2: Element-wise multiplication in the frequency domain
    res_freq = fourier_coef1_freq * fourier_coef2_freq
    
    # Step 3: 2D Inverse Discrete Fourier Transform: transform back from the frequency domain
    res = fft.ifft2(res_freq).real if return_real else fft.ifft2(res_freq)
    
    return res


if __name__ == "__main__":
    # sanity check 1
    fourier_coef1 = fourier_coef2 = torch.arange(1,10).reshape(3,3)
    std = torch.tensor([
        [  1.0000,   4.0000,  10.0000,  12.0000,   9.0000],
        [  8.0000,  26.0000,  56.0000,  54.0000,  36.0000],
        [ 30.0000,  84.0000, 165.0000, 144.0000,  90.0000],
        [ 56.0000, 134.0000, 236.0000, 186.0000, 108.0000],
        [ 49.0000, 112.0000, 190.0000, 144.0000,  81.0000]
    ])
    res = FFT(fourier_coef1, fourier_coef2, return_real=True)
    assert ((torch.abs(res - std) < 1e-4).all()), "Sanity Check #1 Failed!"
    
    # sanity check 2
    fs3, fs4 = torch.rand(5, 5), torch.rand(5, 5)
    std = torch.zeros(9, 9)
    for s1 in range(5):
        for t1 in range(5):
            for s2 in range(5):
                for t2 in range(5):
                    s, t = s1 + s2, t1 + t2
                    std[s, t] += fs3[s1, t1] * fs4[s2, t2]
    res = FFT(fs3, fs4, return_real=True)
    assert ((torch.abs(res - std) < 1e-4).all()), "Sanity Check #2 Failed!"
    
    print("Sanity Check Passed!")