import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


ACTIVATION = {
    'gelu':nn.GELU(), 'tanh':nn.Tanh(), 'sigmoid':nn.Sigmoid(), 'relu':nn.ReLU(), 
    'leaky_relu':nn.LeakyReLU(0.1), 'softplus':nn.Softplus(), 'ELU':nn.ELU(), 'silu':nn.SiLU()
}

# %%
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_num, head_dim, dropout=0.):
        super(SelfAttention, self).__init__()
        assert embed_dim == head_num * head_dim, \
            f"embed_dim {embed_dim} should be divisble by head_num {head_num} and head_dim {head_dim}"
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, D = x.size()
        q = self.linear_q(x).view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k = self.linear_k(x).view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.linear_v(x).view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.einsum("bhic,bhjc->bhij", q, k) * self.scale
        if mask is not None:
            mask = mask.to(x.device)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1) # B H N N
        out = torch.einsum("bhij,bhjc->bhic", attn, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.drop(self.proj(out))
        return out


# %%
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, head_num, head_dim, mlp_ratio=2., act='gelu'):
        super(AttentionBlock, self).__init__()
        self.act = ACTIVATION[act]

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, head_num, head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            self.act,
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# %%
class AFNO1D(nn.Module):
    def __init__(self, width = 32, num_blocks=8, channel_first=False, sparsity_threshold=0.01, modes=32, hidden_size_factor=1, act='gelu'):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = ACTIVATION[act]

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, W = x.shape
            x = x.permute(0, 2, 1)  ### -> N, X, C
        else:
            B, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1,), norm="ortho")

        x = x.reshape(B, x.shape[1], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        ## for ab study
        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], C)
        x = torch.fft.irfft2(x, s=(W,), dim=(1,), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 2, 1)     ### N, C, X

        return x


# %%
class Block(nn.Module):
    def __init__(self, width=32, n_blocks=4, mlp_ratio=1., modes=32, act='gelu', channel_first=True, double_skip=False):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        self.filter = AFNO1D(width, n_blocks, channel_first, modes=modes, act=act)
        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv1d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residual
        return x


# %%
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu'):
        super(PatchEmbed, self).__init__()
        img_size = (img_size[0],)
        patch_size = (patch_size,)
        num_patches = img_size[0] // patch_size[0]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0],)
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv1d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        _, _, W = x.shape
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# %%
class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C

    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)
        return x


# %%
class DPOT1D(nn.Module):
    def __init__(self, args, normalize=False, use_cat=True):
        super(DPOT1D, self).__init__()
        self.in_channels = args.n_channels
        self.out_channels = args.n_channels
        self.in_timesteps = args.T_in
        self.out_timesteps = args.T_bundle
        self.n_blocks = args.n_blocks
        self.modes = args.modes
        self.embed_dim = args.width  # num_features for consistency with other models
        self.mlp_ratio = args.mlp_ratio

        self.act = ACTIVATION[args.act]

        # 
        self.image_size = [int(x) for x in args.img_size.split(',')]
        self.use_cat = use_cat
        if self.use_cat:
            self.patch_embed = PatchEmbed(
                self.image_size, args.patch_size, args.n_channels + 2, args.n_channels * args.patch_size + 2, self.embed_dim, args.act)
        else:
            self.patch_embed = PatchEmbed(
                self.image_size, args.patch_size, args.n_channels, args.n_channels * args.patch_size + 2, self.embed_dim, args.act)
        # 
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.patch_embed.out_size[0]))
        self.normalize = normalize

        # 
        head_dim = args.modes
        head_num = self.embed_dim // head_dim
        self.attn_type = args.mixing_type
        if self.attn_type == 'afno':
            self.blocks = nn.ModuleList([
                Block(width=self.embed_dim, n_blocks=args.n_blocks, mlp_ratio=args.mlp_ratio, modes=args.modes, act=args.act)
                for i in range(args.n_layers)])
        elif self.attn_type == 'standard_sa':
            self.blocks = nn.ModuleList([
                AttentionBlock(self.embed_dim, head_num, head_dim, mlp_ratio=args.mlp_ratio, act=args.act)
                for i in range(args.n_layers)])

        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * args.n_channels, self.embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * args.n_channels, self.embed_dim)

        self.time_agg_layer = TimeAggregator(args.n_channels, args.T_in, self.embed_dim, args.time_agg)

        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels=self.embed_dim, out_channels=args.out_layer_dim, kernel_size=args.patch_size, stride=args.patch_size),
            self.act,
            nn.Conv1d(in_channels=args.out_layer_dim, out_channels=args.out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv1d(in_channels=args.out_layer_dim, out_channels=self.out_channels * self.out_timesteps, kernel_size=1, stride=1)
        )
        # 
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self._init_weights()

    def _init_weights(self,):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.trunc_normal_(m.weight, std=.002)
                if m.bias is not None:
                # if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_grid_temporal(self, x):
        batchsize, size_x, size_t, _ = x.size()
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float) \
            .reshape(1, size_x, 1, 1).to(x.device).repeat([batchsize, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float) \
            .reshape(1, 1, size_t, 1).to(x.device).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridt), dim=-1)
        return grid

    # in/out: B, X, T, C
    def forward(self, x, return_feat=False):
        B, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   #-> B, C, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 3, 1, 2)

        grid = self.get_grid_temporal(x)
        if self.use_cat:
            x = torch.cat((x, grid), dim=-1).contiguous() # B, X, T, C+2
        x = rearrange(x, 'b x t c -> (b t) c x')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x -> b x t c', b=B, t=T)
        x = self.time_agg_layer(x)
        x = rearrange(x, 'b x c -> b c x')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer

        if self.attn_type == 'standard_sa':
            x = rearrange(x, 'b d x -> b x d')

        for blk in self.blocks:
            x = blk(x)

        if self.attn_type == 'standard_sa':
            x = rearrange(x, 'b x d -> b d x', x=self.patch_embed.out_size[0])

        x_out = self.out_layer(x).permute(0, 2, 1)
        x_out = x_out.reshape(*x_out.shape[:2], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x_out = x_out * sigma  + mu

        if return_feat:
            return x_out, x
        return x_out

# %%
def dpot1d(args, normalize=False, use_cat=True):
    model = DPOT1D(args, normalize, use_cat)
    return model
