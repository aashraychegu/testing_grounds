# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:32:24 2021

@author: bjorn

main script to run all training/eval scripts from
"""
import os
os.system("cls")

import copy
import wandb
import torch
import torch.nn as nn
import time
import math
import numpy as np

from CoRe_Dataloader import dataloader,dataset
from TransformerModel import TransformerModel
from train_eval import train, evaluate

train_loader = None


# WandB - Initialize run
wandb.init(project="AF-transformer-test")
# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config  # Initialize config
config.epochs = 1000  # number of epochs to train (default: 10)
config.lr = 1e-2  # learning rate (default: 0.01)
config.log_interval = 1  # how many batches to wait before logging training status
config.emsize = 64  # embedding dimension == d_model
config.dim_feedforward = (
    512  # the dimension of the feedforward network model in nn.TransformerEncoder
)
config.nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
config.nhead = 16  # the number of heads in the multiheadattention models
config.n_conv_layers = 2  # number of convolutional layers (before transformer encoder)
config.dropout = 0.25  # the dropout value
config.dropout_other = 0.1  # dropout value for feedforward output layers
config.n_class = 19
# this is a process
import torch
torch.cuda.empty_cache()

# this is another process
import gc
gc.collect()

# os.system("cls")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # set to gpu if possible


model = TransformerModel(
    config.emsize,
    config.nhead,
    config.dim_feedforward,
    config.nlayers,
    config.n_conv_layers,
    config.n_class,
    config.dropout,
    config.dropout_other,
).to(device)

criterion = nn.CrossEntropyLoss()  # pass logits as input (not probabilities)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, betas=(0.9, 0.98)
)  # weight_decay=1e-6
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.95)


wandb.watch(model, log="all")
history = dict(train=[], val=[], val_acc=[], val_sens=[], val_spec=[])
best_model = torch.nn.Module()
best_val_loss = float("inf")


for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()

    model, train_loss = train(config, model, optimizer, criterion, dataloader, device)
    val_loss, cm, val_acc, val_sens, val_spec = evaluate(
        args=config,
        eval_model=model,
        data_source=dataloader,
        criterion=criterion,
        device=device,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    if epoch > 10:
        scheduler.step()

    epoch_time = time.time() - epoch_start_time
    history["train"].append(train_loss)
    history["val"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_sens"].append(val_sens)
    history["val_spec"].append(val_spec)
    wandb.log({"Epoch Time [s]": epoch_time,"learning_rate":scheduler.get_last_lr()})  # log time for epoch
    print("-" * 89)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}".format(
            epoch, epoch_time, val_loss, math.exp(val_loss)
        )
    )
    print("-" * 89)
    print("valid conf mat:\n", cm)
    print("valid accuracy:", np.round(val_acc, 4))
    print("-" * 89)


torch.save(best_model.state_dict(), "model_save_path.pth")
