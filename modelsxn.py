from modelutils import *
from diff_aug import DiffAugment


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


class Generator(nn.Module):
    """docstring for Generator"""

    # ,device=device):
    def __init__(
        self,
        depth1=5,
        depth2=4,
        depth3=2,
        start_image_side_size=8,
        dim=384,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
        latent_dim=1024,
        output_channels=1,
        psfac=2,
        cvupmult=1,
    ):
        super(Generator, self).__init__()

        self.initial_size = start_image_side_size**2
        self.dim = dim
        self.psfac = psfac
        self.H, self.W = int(self.initial_size**0.5), int(self.initial_size**0.5)
        self.cvupmult = cvupmult
        self.latent_dim = latent_dim
        self.mlp = nn.Linear(latent_dim, (self.initial_size) * self.dim)

        self.encoder_block1 = completeEncoderModule(
            size=self.initial_size,
            dim=dim,
            depth=depth1,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        self.cvup1 = convUp(
            in_channels=(dim // (psfac**2)),
            out_channels=(dim // (psfac**2)) * cvupmult,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.encoder_block2 = completeEncoderModule(
            size=(self.initial_size) * (psfac**2),
            dim=(dim // (psfac**2)) * cvupmult,
            depth=depth2,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        self.cvup2 = convUp(
            in_channels=(dim // (psfac**4)) * cvupmult,
            out_channels=(dim // (psfac**4)) * cvupmult**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.encoder_block3 = completeEncoderModule(
            size=(self.initial_size) * (psfac**4),
            dim=(dim // (psfac**4)) * cvupmult**2,
            depth=depth3,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        self.reshape_for_out = nn.Conv2d(
            (dim // psfac**4) * (cvupmult**2), output_channels, 1, 1, 0
        )

    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initial_size, self.dim)
        # print(x.shape, "initial MLP")

        x = self.encoder_block1(x)
        # print(x.shape, "encoder block 1")

        x, H, W = UpSampling(x, self.H, self.W, psfac=self.psfac)
        # print(x.shape, "upsampling 1")

        x = self.cvup1(x)
        # print(x.shape, "cvup 1")

        x = self.encoder_block2(x)
        # print(x.shape, "encoder block 2")

        x, H, W = UpSampling(x, H, W, psfac=self.psfac)
        # print(x.shape, "upsampling 2")

        x = self.cvup2(x)
        # print(x.shape, "cvup 2")

        x = self.encoder_block3(x)
        # print(x.shape, "encoder block 3")

        x = self.reshape_for_out(
            x.permute(0, 2, 1).view(
                -1, self.dim // (self.psfac**4) * self.cvupmult**2, H, W
            )
        )
        # print(x.shape, "reshaping for output")
        return x

    def generate(self, num_images):
        return self.forward(
            torch.cuda.FloatTensor(
                np.random.normal(0, 1, (num_images, self.latent_dim))
            )
        )
