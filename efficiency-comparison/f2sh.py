# @T.C.: From 2D Fourier bases to spherical harmonics

import torch

def f2sh(fourier_coef, f2sh_bases):
    '''
    From 2D Fourier bases to spherical harmonics.
    Input: 
        fourier_coef: Coefficients of 2D Fourier bases, shape (2L-1, 2L-1)
        f2sh_bases: Precomputed coefficients of spherical harmonics for 2D Fourier bases, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of spherical harmonics, shape (L, 2L-1)
    '''
    
    fourier_coef_t_first = fourier_coef.permute(1, 0)
    sum_positive = (fourier_coef_t_first.unsqueeze(0) * f2sh_bases[:, :, :, 0]).sum(dim=-1) # (L, 2L-1)
    sum_negative = (fourier_coef_t_first.flip(dims=[0]).unsqueeze(0) * f2sh_bases[:, :, :, 1]).sum(dim=-1) # (L, 2L-1)
    res = sum_positive + sum_negative
    return res

def f2sh_channel(fourier_coef, f2sh_bases):
    '''
    From 2D Fourier bases to spherical harmonics (with more than one channels).
    Input: 
        fourier_coef: Coefficients of 2D Fourier bases, shape (C, 2L-1, 2L-1)
        f2sh_bases: Precomputed coefficients of spherical harmonics for 2D Fourier bases, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of spherical harmonics, shape (C, L, 2L-1)
    '''
    
    fourier_coef_t_first = fourier_coef.permute(0, 2, 1)
    sum_positive = (fourier_coef_t_first.unsqueeze(1) * f2sh_bases[:, :, :, 0].unsqueeze(0)).sum(dim=-1) # (C, L, 2L-1)
    sum_negative = (fourier_coef_t_first.flip(dims=[1]).unsqueeze(1) * f2sh_bases[:, :, :, 1].unsqueeze(0)).sum(dim=-1) # (C, L, 2L-1)
    res = sum_positive + sum_negative
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

