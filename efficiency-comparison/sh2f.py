# @T.C.: From spherical harmonics to 2D Fourier bases.

def sh2f(sh_coeff, sh2f_bases):
    '''
    From spherical harmonics to 2D Fourier bases..
    Input: 
        sh_coeff: Coefficients of spherical harmonics, shape (L, 2L-1)
        sh2f_bases: Precomputed coefficients of 2D Fourier bases for spherical harmonics, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of 2D Fourier bases, shape (2L-1, 2L-1)
    '''
    
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * sh2f_bases).sum(dim=0) # (2L-1, 2L-1, 2)
    res = ((sum_along_L[:, :, 0] + sum_along_L[:, :, 1].flip(dims=[0]))).permute(1, 0) # (2L-1, 2L-1)
    return res

def sh2f_channel(sh_coeff, sh2f_bases):
    '''
    From spherical harmonics to 2D Fourier bases (with more than one channels).
    Input: 
        sh_coeff: Coefficients of spherical harmonics, shape (C, L, 2L-1)
        sh2f_bases: Precomputed coefficients of 2D Fourier bases for spherical harmonics, shape (L, 2L-1, 2L-1, 2)
    Output:
        res: Coefficients of 2D Fourier bases, shape (C, 2L-1, 2L-1)
    '''
    
    sum_along_L = (sh_coeff.unsqueeze(-1).unsqueeze(-1) * sh2f_bases.unsqueeze(0)).sum(dim=1) # (C, 2L-1, 2L-1, 2)
    res = ((sum_along_L[:, :, :, 0] + sum_along_L[:, :, :, 1].flip(dims=[1]))).permute(0, 2, 1) # (C, 2L-1, 2L-1)
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

