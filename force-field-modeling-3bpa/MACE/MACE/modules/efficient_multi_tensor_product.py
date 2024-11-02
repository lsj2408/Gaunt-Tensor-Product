import torch
from e3nn import o3
from .efficient_utils import FFT_batch_channel, sh2f_batch_channel, f2sh_batch_channel
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sh2f_bases_dict = torch.load(os.path.join(current_directory,"coefficient_sh2f.pt"))
f2sh_bases_dict = torch.load(os.path.join(current_directory,"coefficient_f2sh.pt"))

class EfficientMultiTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: int,
        device: str,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.dimensions = irreps_in.count((0, 1))
        del irreps_in, irreps_out
        self.correlation = correlation
        self.device = device

        L_in = self.irreps_in.lmax + 1
        L_out = self.irreps_out.lmax + 1
        self.L_in = L_in
        self.L_out = L_out

        self.sh2f_bases = sh2f_bases_dict[L_in].to(device)
        lmaxs = torch.arange(2, correlation + 1) * (L_in - 1)
        self.f2sh_bases_list = list(map(lambda lmax: f2sh_bases_dict[lmax + 1].to(device), lmaxs.tolist()))
        self.offsets_st = lmaxs - L_out + 1
        self.offsets_ed = lmaxs + L_out

        def gen_mask(L):
            left_indices = torch.arange(L, device=device).view(1, -1)  
            right_indices = torch.arange(L - 2, -1, -1, device=device).view(1, -1)  
            column_indices = torch.cat((left_indices, right_indices), dim=1).repeat(L, 1)  
            row_indices = torch.arange(L, device=device).view(-1, 1).repeat(1, 2 * L - 1)  
            mask = torch.abs(column_indices - (L - 1)) <= row_indices  
            mask2D = (torch.ones(L, 2 * L - 1, device=device) * mask).to(bool)
            return mask2D.flatten()
        self.mask_i, self.mask_o = list(map(gen_mask, [L_in, L_out]))

        slices = 2 * torch.arange(L_out, device=device) + 1
        self.slices = slices.tolist()
        
        self.weights = torch.nn.ParameterDict({})
        for i in range(1, correlation + 1):
            w = torch.nn.Parameter(
                torch.randn(1, num_elements, self.dimensions, self.L_out)
            )
            self.weights[str(i)] = w
    
    def forward(self, atom_feat: torch.tensor, atom_type: torch.Tensor):

        # Convert from 3D to 4D, so as to facilitate the implementation of Efficient Gaunt TP
        # The time taken by this convert step is minimal
        n_nodes = atom_feat.shape[0]
        feat3D = torch.zeros(n_nodes, self.dimensions, self.mask_i.shape[0], device=self.device)
        feat3D[:, :, self.mask_i] = atom_feat
        feat4D = feat3D.reshape(n_nodes, self.dimensions, self.L_in, -1) # (B, C, L_in, 2L_in-1)


        # @T.C.: Perform Efficient Gaunt TP
        
        weights = (self.weights["1"] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1) # (B, C, L_out, 1)
        result = feat4D[:, :, :self.L_out, self.L_in-self.L_out:self.L_in+self.L_out-1] * weights
        
        fs_out = {}
        fs_out[1] = sh2f_batch_channel(feat4D, self.sh2f_bases)
        for nu in range(2, self.correlation + 1):
            if nu % 2 == 0:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2])
            else:
                fs_out[nu] = FFT_batch_channel(fs_out[nu//2], fs_out[nu//2 + 1])
            idx = nu - 2
            weights = (self.weights[str(nu)] * atom_type.unsqueeze(-1).unsqueeze(-1)).sum(1).unsqueeze(-1)
            result += weights * f2sh_batch_channel(fs_out[nu], self.f2sh_bases_list[idx]).real[:, :, :self.L_out, self.offsets_st[idx]:self.offsets_ed[idx]]
        

        # Convert from 4D to 2D, so as to match the original codebase
        # The time taken by this convert step is minimal
        if self.L_out == 1:
            return  result.squeeze()
        else:
            result3D_unfiltered = result.reshape(n_nodes, self.dimensions, -1)
            result3D = torch.zeros(n_nodes, self.dimensions, self.mask_o.shape[0], device=self.device)
            result3D = result3D_unfiltered[ :, :, self.mask_o]
            irreps = torch.split(result3D, self.slices, dim=-1)
            irreps_flatten = list(map(lambda x: x.flatten(start_dim=1), irreps))
            result2D = torch.cat(irreps_flatten, dim=-1)
            return result2D