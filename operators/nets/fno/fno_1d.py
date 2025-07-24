import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# %%
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float32)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], torch.view_as_complex(self.weights))

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


# %%
# 1D FNO Model
class FNO1D(nn.Module):
    def __init__(self, args, use_ln=False, normalize=False):
        super(FNO1D, self).__init__()

        self.n_layers = args.n_layers
        self.modes1 = args.modes1
        self.width = args.width

        self.n_channels = args.n_channels
        self.in_timesteps = args.T_in
        self.out_timesteps = args.T_bundle

        self.use_ln = use_ln
        self.normalize = normalize

        # 
        self.padding = [int(x) for x in args.padding.split(',')]
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes1) for _ in range(self.n_layers)
            ])
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_layers)
            ])
        self.act = nn.GELU()
        # 
        if self.use_ln:
            self.ln_layers = nn.ModuleList([
                nn.GroupNorm(4, self.width) for _ in range(self.n_layers)
                ])
        # 
        if self.normalize:
            self.fc_scale = nn.Linear(2 * self.n_channels, self.width)

        self.fc0 = nn.Linear(self.in_timesteps * self.n_channels + 1, self.width)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.n_channels * self.out_timesteps)

    def get_grid(self, x):
        batchsize, size_x = x.shape[0], x.shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float).reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        grid = gridx.to(x.device)
        return grid

    def forward(self, x, return_feat=False):
        if self.normalize:
            mu = x.mean(dim=(1,2), keepdim=True)
            sigma = x.std(dim=(1,2) ,keepdim=True) + 1e-6
            x = (x - mu)/ sigma
            scale_feats = self.fc_scale(torch.cat([mu, sigma], dim=-1)).squeeze(-2)
        else:
            scale_feats = 0.0

        grid = self.get_grid(x)

        x = rearrange(x, 'b x t c -> b x (t c)')
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x) + scale_feats
        x = x.permute(0, 2, 1).contiguous()  ## B, D, X
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2

            if i < (self.n_layers-1):
                x = self.act(x)
                if self.use_ln:
                    x = self.ln_layers[i](x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]
        x_out = x.permute(0, 2, 1)  ## B, X, D
        x_out = self.fc1(x_out)
        x_out = self.act(x_out)
        x_out = self.fc2(x_out)

        x_out = rearrange(x_out, 'b x (t c) -> b x t c', t=self.out_timesteps, c=self.n_channels)
        if self.normalize:
            x_out = x_out * sigma + mu

        if return_feat:
            return x_out, x
        return x_out

# %%
def fno1d(args, use_ln=False, normalize=False):
    model = FNO1D(args, use_ln, normalize)
    return model
