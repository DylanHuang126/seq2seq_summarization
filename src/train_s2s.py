from dataset import Seq2SeqDataset
from utils import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import random
import pickle
import argparse
import json
from torch.utils.data import DataLoader
import numpy as np


class EncoderBiRnn(nn.Module):
    def __init__(self, embedding, hidden_size, dropout=0.5):
        super(EncoderBiRnn, self).__init__()
        input_size = embedding.shape[0]
        embed_size = embedding.shape[1]
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, self.hidden_size, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    def forward(self, inputs):
        #inputs = [seq_len, batch_size]
        
        embedding = self.dropout(self.embedding(inputs))
        
        outputs, hidden = self.gru(embedding)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [seq_len, batch_size, hidden_size * 2]
        #hidden = [batch_size, hidden_size]
        return outputs, hidden
    
class DecoderRnn(nn.Module):
    def __init__(self, embedding, hidden_size, dropout=0.5):
        super(DecoderRnn, self).__init__()
        output_size = embedding.shape[0]
        embed_size = embedding.shape[1]
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, self.hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    
    # decoder's hidden : encoder's last hidden
    def forward(self, inputs, hidden):
        #inputs = [batch_size]
        #hidden = [batch_size, hidden_size]
        
        inputs = inputs.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(inputs))
        
        outputs, hidden = self.gru(embedding, hidden.unsqueeze(0))
        
        out = self.fc(outputs)
        
        #out = [batch_size, output_size]
        return out, hidden.squeeze(0)
    
class Seq2Seq(nn.Module):
    def __init__(self, embedding, hidden_size, max_len, encoder_dropout, decoder_dropout, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.n_words = embedding.shape[0]
        self.encoder = EncoderBiRnn(embedding, hidden_size, encoder_dropout)
        self.decoder = DecoderRnn(embedding, hidden_size, decoder_dropout)
        self.max_len = max_len
        
    def forward(self, inputs, targets):
        batch_size = inputs.shape[1]
        self.max_len = self.max_len if targets is None else targets.shape[0]
        encoder_outputs, hidden = self.encoder(inputs)
        decoder_outputs = torch.zeros(self.max_len, batch_size, self.n_words, device=inputs.device)
        
        input_token = targets[0,:] if targets is not None else torch.ones(batch_size, dtype=torch.long, device=inputs.device)
        
        for t in range(1, self.max_len):
            decoder_output, hidden = self.decoder(input_token, hidden)
            
            decoder_outputs[t] = decoder_output
            top1 = decoder_output.argmax(-1).squeeze(0)
            
            #input_token: always give correct token when training
            input_token = targets[t] if targets is not None else top1 
            
            if torch.sum(input_token != 3) == 0:
                break
                
        return decoder_outputs
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def train(model, train_dataset, val_dataset, args, pt='abstractive.pt'):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                              num_workers=args.workers, shuffle=True, collate_fn=train_dataset.collate_fn)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    
    iters = 0
    min_val_loss = 1000000
    
    for epoch in range(args.epochs):
        tr_loss = 0.
        step_iterator = tqdm(train_loader, desc='Training')
        for step, batch in enumerate(step_iterator):
            iters += 1
            
            model.train()
            text, summ = batch['text'].to(device), batch['summary'].to(device)
            text = text.permute(1,0)
            summ = summ.permute(1,0)
            outputs = model(text, summ)
            outputs = outputs[1:].reshape(-1, outputs.shape[-1]) # [seq_len, n_words]
            summ = summ[1:].reshape(-1) # [seq_len]
            loss = criterion(outputs, summ)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            optimizer.step()
            
            loss = loss.detach().cpu().item()
            tr_loss += loss
              
        val_loss = evaluate(model, val_dataset, criterion, args)
        
        tr_loss /= step + 1
        
        print(f'[Epoch {epoch+1}] loss: {tr_loss:.3f}' + ', ' + f'val_loss: {val_loss:.3f}', flush=True)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), pt)
            
def evaluate(model, dataset, criterion, args):
    dataloader = DataLoader(dataset, batch_size=args.val_batch_size, 
                            num_workers=args.workers, collate_fn=dataset.collate_fn)
    
    model.eval()
    val_loss = 0.
    
    with torch.no_grad():
        step_iterator = tqdm(dataloader, desc='Evaluating')
        for step, batch in enumerate(step_iterator):
            text, summ = batch['text'].to(device), batch['summary'].to(device)
            text = text.permute(1,0)
            summ = summ.permute(1,0)
            outputs = model(text, summ)
            outputs = outputs[1:].reshape(-1, outputs.shape[-1]) # [seq_len, n_words]
            summ = summ[1:].reshape(-1) # [seq_len]
            loss = criterion(outputs, summ)
            
            val_loss += loss.detach().cpu().item()

        val_loss /= step + 1
    
    return val_loss

if __name__ == '__main__':

    with open('../datasets/seq2seq/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../datasets/seq2seq/valid.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('../datasets/seq2seq/embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    args = argparse.Namespace()

    args.train_batch_size = 16
    args.val_batch_size = 16
    args.test_batch_size = 64
    args.embed_size = 300
    args.hidden_size = 256
    args.encoder_dropout = 0.3
    args.decoder_dropout = 0.3
    args.max_len = 80
    args.epochs = 5
    args.clip = 1
    args.lr = 1e-3
    args.workers = 4
    args.seed = 42

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2Seq(embeddings.vectors, args.hidden_size, args.max_len,
                    args.encoder_dropout, args.decoder_dropout).to(device)

    train(model, train_data, val_data, args)

