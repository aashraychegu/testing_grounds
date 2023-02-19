# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:27:48 2021

@author: bjorn

Traininer and Evaluation loops functioning with WandB
"""
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix


def train(args, model, optimizer, criterion, train_loader, device):
    cur_loss = float("inf")
    model.train()  # Turn on the train mode
    total_loss = 0.0
    batch_nr = 0
    start_time = time.time()
    for batch in tqdm(train_loader):
        batch_nr += 1
        data, targets = batch
        data = data.to(torch.float).to(device)
        targets = targets.to(torch.long).to(device)[:, 0]
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        log_interval = 50
        if batch_nr % log_interval == 0 and batch_nr > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            # print('| epoch', epoch,
            #       '| train loss', np.round(cur_loss, 4),
            #       '| ms/batch', np.round(elapsed*1000/log_interval, 3),
            #       '| lr', np.round(scheduler.get_last_lr()[0], 4)
            #       )
            total_loss = 0.0
            start_time = time.time()
        exit()

    # scheduler.step()
    wandb.log({"Train Loss": cur_loss})
    return model, cur_loss


def evaluate(args, eval_model, data_source, criterion, device):
    true_label = np.array([])
    predictions = np.array([])
    outputs = []
    loss_list = []
    eval_model.eval()  # Turn on the evaluation mode
    tot_val_loss = 0.0
    val_batch_nr = len(data_source)
    with torch.no_grad():
        for batch in data_source:
            data, targets = batch
            data, targets = data.to(torch.float).to(device), targets.to(torch.long).to(device)[:,0]
            output = eval_model(data)
            loss = criterion(output, targets)
            tot_val_loss += loss.item()
            preds = torch.argmax(torch.sigmoid(output).cpu().detach(),dim = 1).to(torch.float).numpy()
            predictions = np.append(predictions, preds)
            true_label = np.append(true_label, targets.cpu().detach().to(torch.float))
            for i in (output).cpu().detach().to(torch.float).numpy():
                outputs.append(i)
    # Get losses and accuracy
    # for i,j,a in zip(true_label,predictions,outputs):
        # print(i,j,i == j,a,len(outputs))
    preds = true_label==predictions
    # print(reals)
    goods = np.full(true_label.shape,1)
    cm = confusion_matrix(goods, preds, labels=[0, 1])
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    np.seterr("raise")
    TN, FP, FN, TP = cm.ravel()
    print(f"{TN = }, {FP = }, {FN = }, {TP = },")
    # TPR = TP / (TP + FN)
    # TNR = TN / (TN + FP)
    TPR = 1
    TNR = 0
    wandb.log(
        {
            "Test Accuracy": 100 * acc,
            "Test Sensitivity": 100 * TPR,
            "Test Specificity": 100 * TNR,
            "Test Loss": tot_val_loss / val_batch_nr,
        }
    )
    return tot_val_loss / val_batch_nr, cm, acc, TPR, TNR
