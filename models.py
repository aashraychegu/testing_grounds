import torch
import torch.nn as nn
from diff_aug import DiffAugment
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


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = 1.0 / dim**0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        print("qkv.shape:", qkv.shape, q.shape, k.shape, v.shape)
        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
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


class Generator(nn.Module):
    """docstring for Generator"""

    # ,device=device):
    def __init__(
        self,
        depth1=5,
        depth2=4,
        depth3=2,
        initial_size=8,
        dim=384,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
        latent_dim=1024,
        output_channels=1,
    ):
        super(Generator, self).__init__()

        # self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        self.mlp = nn.Linear(latent_dim, (self.initial_size**2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(
            torch.zeros(1, (initial_size**2), self.dim)
        )
        self.positional_embedding_2 = nn.Parameter(
            torch.zeros(1, (initial_size * 2) ** 2, self.dim // 4)
        )
        self.positional_embedding_3 = nn.Parameter(
            torch.zeros(1, (initial_size * 4) ** 2, self.dim // 16)
        )

        self.TransformerEncoder_encoder1 = TransformerEncoder(
            depth=self.depth1,
            dim=self.dim,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )
        self.TransformerEncoder_encoder2 = TransformerEncoder(
            depth=self.depth2,
            dim=self.dim // 4,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )
        self.TransformerEncoder_encoder3 = TransformerEncoder(
            depth=self.depth3,
            dim=self.dim // 16,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.droprate_rate,
        )

        self.linear = nn.Sequential(nn.Conv2d(self.dim // 16, output_channels, 1, 1, 0))

    def forward(self, noise):
        H, W = self.initial_size, self.initial_size
        x = self.mlp(noise).view(-1, self.initial_size**2, self.dim)
        # print(x.shape, 1)
        x = x + self.positional_embedding_1
        # print(x.shape, 2)
        x = self.TransformerEncoder_encoder1(x)
        # print(x.shape, 3)
        x, H, W = UpSampling(x, H, W)
        # print(x.shape, 4)
        x = x + self.positional_embedding_2
        # print(x.shape, 5)
        x = self.TransformerEncoder_encoder2(x)
        # print(x.shape, 6)
        x, H, W = UpSampling(x, H, W)
        # print(x.shape, 7)
        x = x + self.positional_embedding_3
        # print(x.shape, 8)
        x = self.TransformerEncoder_encoder3(x)
        # print(x.shape, 9)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 16, H, W))
        # print(x.shape, 10)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        diff_aug,
        image_size=32,
        patch_size=4,
        input_channel=3,
        num_classes=1,
        dim=384,
        depth=7,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(
            depth, dim, heads, mlp_ratio, drop_rate
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)
    generator = Generator(
        depth1=1,
        depth2=1,
        depth3=1,
        initial_size=16,
        dim=128 * 4,
        heads=4,
        mlp_ratio=2,
        drop_rate=0.5,
        latent_dim=256,
    ).to(device)

    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 256)))
    noise = noise.to(device)
    print(generator(noise).shape)
