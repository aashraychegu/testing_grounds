# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:20:16 2021

@author: bjorn

script for transformer model
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(1), :].squeeze(1)
        x = x + self.pe[: x.size(0), :]  # type: ignore        
        return x


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        print(f"{batch_rep.shape = }")
        a = self.W(batch_rep)
        print(f"{a.shape = }")
        softmax = nn.functional.softmax
        att_w = softmax(a.squeeze(-1), -1)
        print(f"{att_w.shape = }")
        b = att_w.unsqueeze(-1)
        print(f"{b.shape = }")
        utter_rep = torch.sum(batch_rep * b, dim=1)
        print(f"{utter_rep.shape = }")
        return utter_rep


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        nlayers,
        n_conv_layers=2,
        n_class=2,
        dropout=0.5,
        dropout_other=0.1,
    ):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.relu = torch.nn.ReLU()
        self.pos_encoder = PositionalEncoding(1168, dropout)
        self.self_att_pool = SelfAttentionPooling(d_model)

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.flatten_layer = torch.nn.Flatten()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_other),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_other),
            nn.Linear(d_model, 512),
            torch.nn.Flatten(),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(512, 19)
        )
        self.conv1 = torch.nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32, out_channels=d_model, kernel_size=3, stride=1, padding=1
        )
        self.conv = torch.nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.1)
    def forward(self, src):
        # size input: [batch, sequence, embedding dim.]
        # Resize to --> [batch, input_channels, signal_length]
        src = src.view(-1, 1, src.shape[1])
        src = self.relu(self.conv1(src))
        src = self.relu(self.conv2(src))
        for i in range(self.n_conv_layers):
            src = self.relu(self.conv(src))
            src = self.maxpool(src)
        src = self.pos_encoder(src)
        # print(src.shape) # [batch, embedding, sequence]
        # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        src = src.permute(2, 0, 1)
        # print('src shape:', src.shape)
        # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = self.self_att_pool(output)
        return nn.functional.softmax(self.decoder(output))
