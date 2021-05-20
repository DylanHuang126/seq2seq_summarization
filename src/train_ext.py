from dataset import SeqTaggingDataset
from utils import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import pickle
import argparse
import numpy as np
import json

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_cls, embedding_weight, dropout=0.2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, n_cls)
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)  # lstm (seq_len, batch_size, hidden_size * 2)
        out = torch.stack([self.fc(x[t]) for t in range(x.size(0))])
        return out
    
def train(model, train_data, val_data, args, pt='extractive.pt'):
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size,
                              num_workers=args.workers, collate_fn=train_data.collate_fn, shuffle=True)
    
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    total_step = len(train_loader)
    
    tr_loss = 0.
    min_val_loss = 10000
    
    iters = 0
    pos_weight = torch.tensor([args.pos_weight], device=device)
    for epoch in range(args.epochs):
        tr_loss = 0.
        step_iterator = tqdm(train_loader, desc="Training")
        
        for idx, batch in enumerate(step_iterator):
            iters += 1
            
            text = batch['text'].to(device)
            label = batch['label'].type(torch.FloatTensor).to(device)
            label_len = batch['label_len']
            weight = torch.zeros_like(label)
            
            for idx, r in enumerate(weight):
                r[:label_len[idx]] = 1

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight)
            
            model.train()
            outputs = model(text).squeeze(-1)
            loss = criterion(outputs, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss = loss.detach().cpu().item()
            
            tr_loss += loss
        tr_loss /= len(train_loader)
        val_loss = evaluate(model, val_data, args)
        
        print(f'[Epoch {epoch+1}] loss: {tr_loss:.3f} '+
              f'val_loss: {val_loss:.3f}', flush=True)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), pt)
            
def evaluate(model, val_data, args, verbose=False, return_pred=False):
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size,
                            num_workers=args.workers, collate_fn=val_data.collate_fn)
    
    val_loss = 0.
    total_steps = 0
    prediction = []
    pos_weight=torch.tensor([args.pos_weight], device=device)
    step_iterator = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for idx, batch in enumerate(step_iterator):
            text = batch['text'].to(device)
            label = batch['label'].type(torch.FloatTensor).to(device)
            label_len = batch['label_len']
            
            outputs = model(text).squeeze(-1)
            
            weight = torch.zeros_like(label)
            for idx, r in enumerate(weight):
                r[:label_len[idx]] = 1

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight)
            
            loss = criterion(outputs, label)
            val_loss += loss.detach().cpu().item()
            
    return val_loss / len(step_iterator)

if __name__ == '__main__':
    
    with open('../datasets/seq_tag/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../datasets/seq_tag/valid.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('../datasets/seq_tag/embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        
    args = argparse.Namespace()
    args.train_batch_size = 16
    args.val_batch_size = 128
    args.test_batch_size = 1
    args.epochs = 5
    args.lr = 1e-3
    args.workers = 4
    args.seed = 42
    args.n_cls = 1
    args.n_layers = 2
    args.input_size = 300
    args.hidden_size = 128
    args.pos_weight = 4.3
    args.ignore_index = -100
    args.dropout = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                 n_layers=args.n_layers,n_cls=args.n_cls,embedding_weight=embeddings.vectors).to(device)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def plot_rel_loc(all_ext, all_n_sent, bins=20):
        rel_loc = []
        for e, n in zip(all_ext, all_n_sent):
            rel_loc.append(np.array(e) / n)
        rel_loc = np.array(rel_loc)
        x = [i for rel in rel_loc for i in rel]
        plt.hist(x, bins=bins, range=(0,1))
        plt.xlabel("relative location")
        plt.ylabel("density")
        plt.title("distribution of relative location")
        plt.show()

    set_seed(args.seed)

    train(model, train_data, val_data, args, 'model_ext.pt')

