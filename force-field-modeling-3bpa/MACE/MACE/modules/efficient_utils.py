import torch
import torch.fft as fft

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


def sh2f_batch_channel(sh_coeff, sh2f_bases):
    '''
    From spherical harmonics to 2D Fourier bases (with more than one batches and channels).
    Input: 
        sh_coeff: Coefficients of spherical harmonics, shape (B, C, L, 2L-1)
        sh2f_bases: Precomputed coefficients of 2D Fourier bases for spherical harmonics, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of 2D Fourier bases, shape (B, C, 2L-1, 2L-1)
    '''
    
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * sh2f_bases.unsqueeze(0).unsqueeze(0)).sum(dim=2) # (B, C, 2L-1, 2L-1, 2)
    res = ((sum_along_L[:, :, :, :, 0] + sum_along_L[:, :, :, :, 1].flip(dims=[2]))).permute(0, 1, 3, 2) # (B, C, 2L-1, 2L-1)
    return res


def f2sh_batch_channel(fourier_coef, f2sh_bases):
    '''
    From 2D Fourier bases to spherical harmonics (with more than one batches and channels).
    Input: 
        fourier_coef: Coefficients of 2D Fourier bases, shape (B, C, 2L-1, 2L-1)
        f2sh_bases: Precomputed coefficients of spherical harmonics for 2D Fourier bases, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of spherical harmonics, shape (B, C, L, 2L-1)
    '''
    
    fourier_coef_t_first = fourier_coef.permute(0, 1, 3, 2)
    sum_positive = (fourier_coef_t_first.unsqueeze(2) * f2sh_bases[:, :, :, 0].unsqueeze(0).unsqueeze(0)).sum(dim=-1) # (B, C, L, 2L-1)
    sum_negative = (fourier_coef_t_first.flip(dims=[2]).unsqueeze(2) * f2sh_bases[:, :, :, 1].unsqueeze(0).unsqueeze(0)).sum(dim=-1) # (B, C, L, 2L-1)
    res = sum_positive + sum_negative
    return res
