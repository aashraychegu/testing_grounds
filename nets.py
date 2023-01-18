import torch
from torch import nn
import torch.nn.functional as F

maxlen, numclasses = 40817, 19


class convclassifier(nn.Module):
    def __init__(self, input_length=maxlen):
        super().__init__()
        # self.inputnorm = nn.InstanceNorm1d(input_length, affine=True)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.gn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=2,
            dilation=1,
        )
        self.gn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=2,
            dilation=1,
        )
        self.gn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.gn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(
            in_channels=512,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
        )
        self.gn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(
            in_channels=256,
            out_channels=128,
            kernel_size=6,
            stride=4,
            padding=1,
            dilation=1,
        )
        self.gn6 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=3,
            padding=1,
            dilation=1,
        )
        self.gn7 = nn.BatchNorm1d(64)
        self.conv8 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=3,
            padding=2,
            dilation=1,
        )
        self.gn8 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(1152, 512)
        self.gn9 = nn.BatchNorm1d(1)
        self.output = nn.Linear(512, numclasses)
        self.activation = nn.GELU()

    def forward(self, inp):
        # inp = self.inputnorm(inp)
        inp = self.activation(self.gn1(self.conv1(inp)))
        inp = self.activation(self.gn2(self.conv2(inp)))
        inp = self.activation(self.gn3(self.conv3(inp)))
        inp = self.activation(self.gn4(self.conv4(inp)))
        inp = self.activation(self.gn5(self.conv5(inp)))
        inp = self.activation(self.gn6(self.conv6(inp)))
        inp = self.activation(self.gn7(self.conv7(inp)))
        inp = self.activation(self.gn8(self.conv8(inp)))
        inp = inp.view(1, 1, -1)
        inp = self.activation(self.gn9(self.linear1(inp)))
        inp = self.activation(self.output(inp))
        return inp


cc = convclassifier()
inp = torch.randn(1, 1, 40817)
out = cc.forward(inp)
print(out.shape,out[0][0])
