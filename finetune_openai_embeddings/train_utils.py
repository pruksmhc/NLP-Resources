import os
import random
import time

import openai
import pickle
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F
import regex as re
import torch.nn as nn
import torch.optim as optim

from openai.embeddings_utils import get_embedding
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List
import tqdm

openai.api_key = ''

def calculate_rpa(y_true, y_pred):    
    
    y_pred = y_pred.ravel()
    
    t0 = time.time()
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    t1 = time.time()
    
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    true_negative = ((y_true == 0) & (y_pred == 0)).sum()

    recall = true_positive / (true_positive + false_negative + 1e-6)
    precision = true_positive / (true_positive + false_positive + + 1e-6)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-6)
    
    return recall, precision, accuracy

def _evaluation(writer, test_dataloader, device, model, epoch, pos_weight):
    model.eval()
    total = 0
    correct = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        eval_loss = 0.0
        for batch_idx, (user_emb, item_emb, label) in tqdm.tqdm(enumerate(test_dataloader)):
            user_emb, item_emb, label = user_emb.to(device), item_emb.to(device), label.to(device)
            outputs = torch.sigmoid(model(user_emb, item_emb))
            predicted_labels = (outputs > 0.5).float()
            y_pred.append((outputs > 0.5).float())            
            total += label.size(0)
            correct += (predicted_labels == label.float().unsqueeze(1)).sum().item()
            y_true.append(np.ravel(label.detach().cpu().numpy()))
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, label.float().unsqueeze(1))
            eval_loss += loss.item()
    
    eval_loss /= len(test_dataloader)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate([each.cpu() for each in y_pred])
    
    auc = roc_auc_score(y_true, y_pred)  
    recall, precision, accuracy = calculate_rpa(y_true, y_pred)
    writer.add_scalar("Loss/Epoch(test)", eval_loss, epoch)    
    writer.add_scalar("Recall/Epoch(test)", recall, epoch)
    writer.add_scalar("Precision/Epoch(test)", precision, epoch)
    writer.add_scalar("Accuracy/Epoch(test)", accuracy, epoch)
    writer.add_scalar("Auc/Epoch(test)", auc, epoch)
    
    return eval_loss, recall, precision, accuracy, auc

def _train(train_dataloader, device, model, optimizer, epoch, save_path, writer, pos_weight):
    model.train()
    train_loss = 0.0
    for batch_idx, (user_emb, item_emb, label) in tqdm.tqdm(enumerate(train_dataloader)):     
        user_emb, item_emb, label = user_emb.to(device), item_emb.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = torch.sigmoid(model(user_emb, item_emb))
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, label.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/step(train)", loss, batch_idx + epoch * len(train_dataloader))
        train_loss += loss.item()

    train_loss = train_loss/len(train_dataloader)
    _, recall, precision, accuracy, auc = _evaluation(writer, train_dataloader, device, model, epoch, pos_weight)

    # print(f"   Training accuracy is {accuracy:.4f}, AUC is {auc:.4f}")
    writer.add_scalar("Loss/Epoch(train)", train_loss, epoch)
    writer.add_scalar("Recall/Epoch(train)", recall, epoch)
    writer.add_scalar("Precision/Epoch(train)", precision, epoch)
    writer.add_scalar("Accuracy/Epoch(train)", accuracy, epoch)
    writer.add_scalar("Auc/Epoch(train)", auc, epoch)
    
    model_path = os.path.join(save_path, f"{epoch}-model.pt")
    torch.save(model.state_dict(), model_path)
    
    return train_loss, recall, precision, accuracy, auc

def train(train_dataloader, test_dataloader, device, model, num_epochs=200, ratio_pos_neg=99):
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([float(ratio_pos_neg)]).to(device)
    writer = SummaryWriter()
    for epoch in range(num_epochs):    
        train_loss, train_recall, train_precision, train_accuracy, train_auc = _train(train_dataloader, device, model, optimizer, epoch, '.', writer, pos_weight)
        eval_loss, eval_recall, eval_precision, eval_accuracy, eval_auc = _evaluation(writer, test_dataloader, device, model, epoch, pos_weight) 
        table = {"Epoch": [epoch],
                "train_loss": [f"{train_loss:.4f}"],
                "eval_loss": [f"{eval_loss:.4f}"],
                "train_recall": [f"{train_recall:.4f}"],
                "eval_recall": [f"{eval_recall:.4f}"],
                "train_precision": [f"{train_precision:.4f}"],
                "eval_precision": [f"{eval_precision:.4f}"],
                "train_accuracy": [f"{train_accuracy:.4f}"],
                "eval_accuracy": [f"{eval_accuracy:.4f}"],
                "train_auc": [f"{train_auc:.4f}"], 
                "eval_auc": [f"{eval_auc:.4f}"]}

        print(tabulate(table, headers="keys"))
            
    writer.close()
