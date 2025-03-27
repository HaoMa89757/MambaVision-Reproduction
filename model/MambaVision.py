import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch.nn import Conv2d, BatchNorm2d, GELU
from timm.models.layers import DropPath, trunc_normal_, LayerNorm2d
from timm.models.vision_transformer import Mlp


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, H//ws, W//ws, ws, ws, C)
    x = x.view(-1, window_size * window_size, C)
    # print("window_partition:x.shape", x.shape)
    return x


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size*window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    # print("window_reverse:x.shape", x.shape)
    return x


class Patch_Embedding(nn.Module):
    def __init__(self, in_channels=3, inner_dim=32, dim=80):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, inner_dim, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(inner_dim, eps=1e-4),
            nn.ReLU(),
            Conv2d(inner_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )
        # print("Patch_Embedding:conv", self.conv)

    def forward(self, x):
        x = self.conv(x)
        # print("Patch_Embedding:x.shape", x.shape)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, drop_path=0.):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            BatchNorm2d(dim, eps=1e-5),
            GELU(approximate='tanh'),
            Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            BatchNorm2d(dim, eps=1e-5)
        )
        self.proj = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print("ConvBlock:conv", self.conv)
        # print("ConvBlock:proj", self.proj)
        # print("ConvBlock:drop_path", self.drop_path)

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv(x)
        x = residual + self.drop_path(x)
        # print("ConvBlock:x.shape", x.shape)
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False),
        )
        # print("Downsample:conv", self.conv)

    def forward(self, x):
        x = self.conv(x)
        # print("Downsample:x.shape", x.shape)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # print("Attention:num_heads", self.num_heads)
        # print("Attention:head_dim", self.head_dim)
        # print("Attention:scale", self.scale)
        # print("Attention:qkv", self.qkv)
        # print("Attention:norm", self.norm)
        # print("Attention:attn_drop", self.attn_drop)
        # print("Attention:proj", self.proj)
        # print("Attention:proj_drop", self.proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # x: (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("Attention:x.shape", x.shape)
        return x

class MambaVision_Mixer(nn.Module):
    def __init__(self, d_model, d_state=32, dt_scale=1.0, dt_max=0.1, dt_min=0.001, dt_init_floor=1e-4, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = 1
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner, )

        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner // 2, ) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()

        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=False,
            kernel_size=d_conv,
            groups=self.d_inner // 2
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=False,
            kernel_size=d_conv,
            groups=self.d_inner // 2
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class MixerBlock(nn.Module):
    def __init__(self, idx, num, dim, drop_path=0., mlp_ratio=4, drop=0., layer_scale=None, num_heads=8):
        super().__init__()
        # 前半部分使用 MambaVision_Mixer，否则使用 Attention
        if idx < (num // 2 + num % 2):
            self.mixer = MambaVision_Mixer(d_model=dim, d_state=8)
        else:
            self.mixer = Attention(dim, num_heads=num_heads, proj_drop=drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm_1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm_2(x)))
        # print("MixerBlock:x.shape", x.shape)
        return x


# 整体模型，支持传入更多参数（如 num_heads, mlp_ratio, drop_path_rate, resolution）
class MambaVision(nn.Module):
    def __init__(self,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 dim=80,
                 in_dim=96,
                 mlp_ratio=4,
                 drop_path_rate=0.2,
                 num_classes=100):
        super().__init__()
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.num_classes = num_classes
        self.patch_embed = Patch_Embedding(in_channels=3, inner_dim=in_dim, dim=dim)
        num_features = int(dim * 2 ** (len(depths) - 1))

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_dim = dim

        block_id = 0
        current_block = 0  # 全局块计数器
        # 根据各阶段深度构造模块
        for i, stage_depth in enumerate(depths):
            self.window_size = window_size[i]
            stage_dpr = dpr[current_block:current_block + stage_depth]
            current_block += stage_depth  # 更新全局计数器
            stage_blocks = []
            for j in range(stage_depth):
                if i < 2:
                    stage_blocks.append(ConvBlock(current_dim, drop_path=stage_dpr[j]))
                else:
                    stage_blocks.append(
                        MixerBlock(
                            idx=j,
                            num=stage_depth,
                            dim=current_dim,
                            drop_path=stage_dpr[j],
                            mlp_ratio=mlp_ratio,
                            drop=0.,
                            layer_scale=None,
                            num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads
                        )
                    )
                block_id += 1
            self.stages.append(nn.Sequential(*stage_blocks))
            if i < len(depths) - 1:
                self.downsamples.append(Downsample(current_dim))
                current_dim *= 2

        # 分类头
        self.norm = nn.BatchNorm2d(num_features)
        # print("num_feature", num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def add_op(self, x, op_type):
        global Hp, Wp, pad_r, pad_b,H,W
        if op_type == 1:
            _, _, H, W = x.shape
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
        else:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

        return x


    def feature_extraction(self, x):
        x = self.patch_embed(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i == 2 or i == 3:
                x = self.add_op(x, op_type=2)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

            if i == 1 or i == 2:
                x = self.add_op(x, op_type=1)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.head(x)
        return x
