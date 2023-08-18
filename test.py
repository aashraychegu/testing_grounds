#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from CoRe_Dataloader_From_Files_With_OvGN import (
    get_new_train_validation_test_datasets,
    get_new_train_validation_test_dataloaders,
)

# from CoRe_Dataloader_From_File_With_Random_From_Tensors import (
#   get_new_train_validation_test_datasets,
#  get_new_train_validation_test_dataloaders,
# )
# from CoRe_Dataloader_ECSG import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import torchinfo
import time
import numpy as np
import wandb
import datetime
from collections import OrderedDict
from torch import autograd
import matplotlib.pyplot as plt

cuda_device = "cuda:0"

import torchmetrics as metrics
import pandas as pd

mae = metrics.MeanAbsoluteError()
mse = metrics.MeanSquaredError()
combined = metrics.MetricCollection(
    [mae, mse, metrics.MeanAbsolutePercentageError(), metrics.MeanSquaredLogError()]
)


def get_df_from_rdict(rdict):
    return pd.DataFrame(pd.Series(rdict).map(lambda x: x.cpu().item())).T


from vit_pytorch import vit_for_small_dataset as vit_sd
from vit_pytorch import vit as simple_vit
from vit_pytorch.deepvit import DeepViT


def init_model():
    # return simple_vit.ViT(image_size=400,
    #                patch_size=20,
    #                num_classes=19,
    #                dim=int(1024/2),
    #                depth=2,
    #                heads=8,
    #                mlp_dim=int(2048/2),
    #                channels=1).to(cuda_device)
    # return vit_sd.ViT(image_size=400,
    #                patch_size=20,
    #                num_classes=19,
    #                dim=1024,
    #                depth=4,
    #                heads=16,
    #                mlp_dim=int(2048/2),
    #                dropout = 0.1,
    #                emb_dropout = 0,
    #                channels=1).to(cuda_device)
    return DeepViT(
        image_size=400,
        patch_size=20,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=20,
        mlp_dim=int(2048 / 2),
        dropout=0.1,
        emb_dropout=0.1,
        channels=1,
    ).to(cuda_device)


# In[ ]:


dumstring = " "


def calc_metrics(model: torch.nn.Module, dl: DataLoader):
    model.eval()
    raw_output = []
    parameters = []
    with torch.no_grad():
        for batch, (sg, params) in enumerate(dl):
            sg = sg.to(cuda_device).float()
            sgsh = sg.shape
            sg = sg.view(sgsh[0], 1, sgsh[1], sgsh[2])

            params = params[:, 1:3].to(cuda_device)
            raw_output.append(model(sg).detach().cpu())
            parameters.append(params.cpu())
            print(f"{batch+1} / {len(dl)} { dumstring  * 200 }", end="\r")
    model.train()
    output = torch.vstack(raw_output)
    parameters = torch.concat(parameters, dim=0)
    return combined(output.cpu(), parameters.cpu())


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


model = init_model()
startlr = 3e-5
optimizer = optim.AdamW(params=model.parameters(), lr=startlr)
optimizer1 = optim.NAdam(params=model.parameters(), lr=startlr)
step_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[1, 2, 3, 4], gamma=0.5
)
# at the end of 600 epochs, the learning rate is 0.000,002,62
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
scheduler_pl = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="max", factor=0.7, patience=35, verbose=True
)
l1 = nn.L1Loss(reduction="sum")
l2 = nn.MSELoss(reduction="sum")
lcosh = LogCoshLoss()
ltanh = XTanhLoss()
xsig = XSigmoidLoss()
lossfn = lambda x, y: l1(x, y) + l2(x, y) + lcosh(x, y) + ltanh(x, y) + xsig(x, y)


def to_seconds(s):
    return f"{s//3600}H:{(s%3600)//60}M:{round(s%60,3)}S"


def ismult(n, div):
    return bool(1 >> (n % div))


def save_model(best_model, config, m1, m2, m1name="l1", m2name="l2"):
    name = f"./for_ligo_cluster/saved_models/ViT/WithNoise/best_model_state_dict_ViT_regressor_for{config.run_name}_stime_{config.start_time.replace(':', '-')}__{m1name}_{m1}__{m2name}_{m2}.pt"
    try:
        torch.save(
            best_model,
            name,
        )
        print("\nSAVING MODEL")
        wandb.save(name)
    except:
        wandb.alert(level="warning", title="OUT OF MEMORY")


def train_eval_model(config, train_dl, test_dl, adam=True, nadam=False):
    min_mae, min_mse = float("inf"), float("inf")
    ldl = len(train_dl)
    results = pd.DataFrame()
    best_model = OrderedDict()
    for epoch in range(1, config.epochs + 1):
        print("Pre-Evaluation Finished; Starting Training")
        etime = time.time()
        for batch, (sg, params) in enumerate(train_dl):
            stime = time.time()
            sgsh = sg.shape
            sg = sg.to(cuda_device).to(torch.float).view(sgsh[0], 1, sgsh[1], sgsh[2])
            params = params[:, 1:3].to(cuda_device).to(torch.float)
            optimizer.zero_grad()
            outputs = model(sg)
            loss = lossfn(outputs, params)
            loss.backward()
            optimizer.step() if adam else None
            optimizer1.step() if nadam else None
            #
            torch.cuda.empty_cache()
            #
            wandb.log(
                {
                    "loss": loss.item(),
                    "batch_mae": mae(outputs.to("cpu"), params.to("cpu")),
                    "batch_mse": mse(outputs.to("cpu"), params.to("cpu")),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "batch": batch,
                }
            )

            print(
                f"{epoch:5}/{config.epochs:5} // {batch:5}/{ldl:5} | Loss: {loss.item():2.4},batch_mae:{mae(outputs.to('cpu'),params.to('cpu')):3.4}, lr:{scheduler.get_last_lr()[0]:1.5}, Time per Batch: {time.time()-stime:.3} seconds, Accumulated Time {to_seconds(round(time.time()-etime,3))}    ",
                end="\r",
                flush=True,
            )

            if (batch - 1) % 1000 == 0:
                epoch_results = calc_metrics(model, test_dl)
                results = pd.concat([results, get_df_from_rdict(epoch_results)])
                min_mae = min(results["MeanAbsoluteError"])
                min_mse = min(results["MeanSquaredError"])
                #
                if epoch_results["MeanAbsoluteError"] <= min_mae:
                    best_model = model.state_dict()
                    save_model(
                        best_model,
                        config,
                        list(epoch_results.values())[0],
                        list(epoch_results.values())[1],
                    )

                wandb.log(
                    {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
                    | epoch_results
                    | {"MinimumMAE": min_mae, "MinimumMSE": min_mse}
                    | {"EpochTime": time.time() - etime}
                )
        #
        epoch_results = calc_metrics(model, test_dl)
        results = pd.concat([results, get_df_from_rdict(epoch_results)])
        #
        min_mae = min(results["MeanAbsoluteError"])
        min_mse = min(results["MeanSquaredError"])
        #
        scheduler.step()
        step_scheduler.step()
        scheduler_pl.step(min_mae)

        if epoch_results["MeanAbsoluteError"] <= min_mae:
            best_model = model.state_dict()
            save_model(
                best_model,
                config,
                list(epoch_results.values())[0],
                list(epoch_results.values())[1],
            )
        #

        wandb.log(
            {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
            | epoch_results
            | {"MinimumMAE": min_mae, "MinimumMSE": min_mse}
            | {"EpochTime": time.time() - etime}
        )

    epoch_results = calc_metrics(model, test_dl)
    results = pd.concat([results, get_df_from_rdict(epoch_results)])
    return min_mae, min_mse


# uncomment for training
results = []
trials = 1
for i in range(trials):
    wandb.init(
        project="ldas-test",
    )
    config = wandb.config
    config.run_name = wandb.run._run_id
    config = wandb.config
    config.epochs = 5
    config.inx = 400
    config.iny = 400
    config.lr = startlr
    config.trial = i + 1
    config.total_trials = trials
    config.best_model = OrderedDict()
    config.start_time = datetime.datetime.now().isoformat()
    config.savename = f"best_model_state_dict_at_for{config.run_name}_stime_{config.start_time.replace(':', '-')}__acc_max_acc__auc_auc.pt"
    train_dl, valid_dl, test_dl = get_new_train_validation_test_dataloaders(
        device=cuda_device
    )
    train_eval_model(wandb.config, train_dl, valid_dl, nadam=True)
    results.append(calc_metrics(model, test_dl))  # type: ignore
    if i != (trials - 1):
        model = init_model()

torch.save(
    model.state_dict(),
    f"./for_ligo_cluster/saved_models/ViT/WithNoise/best_model_state_dict_ViT_regressor_11_07_2023",
)

evaldl = test_dl

model.eval()
totout = []
with torch.no_grad():
    for batch, (sg, params) in enumerate(evaldl):
        sg = sg.to(cuda_device).to(torch.float)
        sgsh = sg.shape
        sg = sg.view(sgsh[0], 1, sgsh[1], sgsh[2])
        modelout = model(sg).detach().cpu()
        params = params.to("cpu").to(torch.float)
        comb = torch.concat([modelout, params], dim=1)
        print(comb[1], modelout[1], params[1])
        print(batch, "finished")
        totout.append(comb)
model.train()
print(len(totout))

all_params = torch.cat(totout)

import pandas as pd

df = pd.DataFrame(all_params.numpy())
df = df.rename(
    columns={0: "PM1", 1: "PM2", 2: "EOS", 3: "M1", 4: "M2", 5: "SHFT", 6: "SNR"}
)
df["combined"] = df["M1"] + df["M2"]
df["DiffM1"] = abs(df["M1"] - df["PM1"])
df["DiffM2"] = abs(df["M2"] - df["PM2"])
df["totDiff"] = df["DiffM1"] + df["DiffM2"]
df["avgDiff"] = df["totDiff"] / 2
df.to_csv("results.csv")

wandb.save("results.csv")
wandb.finish(60)
