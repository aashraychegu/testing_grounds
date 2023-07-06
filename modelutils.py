import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.0):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            input_channel, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim = dim
        # print(dim)
        self.scale = 1.0 / dim**0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # print("qkv.shape:", qkv.shape, "Dim:", self.dim, x.shape)
        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


def UpSampling(x, H, W, psfac=2):
    B, N, C = x.size()
    # assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(psfac)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.0):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList(
            [Encoder_Block(dim, heads, mlp_ratio, drop_rate) for i in range(depth)]
        )

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x


def to_square(x: torch.Tensor):
    B, N, C = x.size()
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, int(N**0.5), int(N**0.5))
    return x


def to_flat(x: torch.Tensor):
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x


class convUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        print(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = to_square(x)
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.act(x)
        x = to_flat(x)
        return x


class completeEncoderModule(nn.Module):
    def __init__(self, size, dim, depth, heads, mlp_ratio, drop_rate):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(size=(1, size, dim)))
        self.tfenc = TransformerEncoder(
            depth=depth,
            dim=dim,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        x = x + self.positional_embedding
        x = self.tfenc(x)
        return x
