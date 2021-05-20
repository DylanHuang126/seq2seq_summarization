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
import numpy as np
from torch.utils.data import DataLoader


class EncoderBiRnn(nn.Module):
    def __init__(self, embedding, hidden_size, dropout=0.5):
        super(EncoderBiRnn, self).__init__()
        self.input_size = embedding.shape[0]
        self.embed_size = embedding.shape[1]
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    def forward(self, inputs):
        #inputs = [seq_len, batch_size]
        
        embedding = self.dropout(self.embedding(inputs))
        #embedding = [seq_len, batch size, embed_size]
        
        outputs, hidden = self.gru(embedding)
        #outputs = [seq len, batch_size, hidden_size * 2], hidden = [1, batch_size, hidden_size]
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [seq_len, batch_size, hidden_size * 2]
        #hidden = [batch_size, hidden_size]
        return outputs, hidden    
    
    
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hidden_size * 2) + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch_size, dec_hidden_size]
        #encoder_outputs = [seq_len, batch_size, enc_hidden_size * 2]
        
        batch_size = encoder_outputs.shape[1]
        seq_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        #hidden = [batch_size, seq_len, dec_hidden_size]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, seq_len, enc_hidden_size * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch_size, seq_len, dec_hidden_size]

        attention = self.v(energy).squeeze(2)
        #attention= [batch_size, seq_len]
        
        return F.softmax(attention, dim=1)
    
class DecoderRnn(nn.Module):
    def __init__(self, embedding, enc_hidden_size, dec_hidden_size, attention, dropout=0.5):
        super(DecoderRnn, self).__init__()
        self.output_size = embedding.shape[0]
        self.embed_size = embedding.shape[1]
        
        self.attention = attention
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU((enc_hidden_size * 2) + self.embed_size, dec_hidden_size)
        self.fc = nn.Linear((enc_hidden_size * 2) + dec_hidden_size, self.output_size)
        
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    def forward(self, inputs, hidden, encoder_outputs):     
        #input = [batch_size]
        #hidden = [batch_size, dec_hidden_size]
        #encoder_outputs = [seq_len, batch_size, enc_hidden_size * 2]
        inputs = inputs.unsqueeze(0)
        
        #input = [1, batch_size]
        
        embedded = self.dropout(self.embedding(inputs))
        #embedded = [1, batch_size, embed_size]
        
        attn_weight = self.attention(hidden, encoder_outputs).unsqueeze(1)
        #attn_weight = [batch_size, 1, seq_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, seq_len, enc_hidden_size * 2]
        
        context = torch.bmm(attn_weight, encoder_outputs)
        #context = [batch_size, 1, enc_hidden_size * 2]
        
        context = context.permute(1, 0, 2)
        #context = [1, batch_size, enc_hidden_size * 2]
        
        gru_input = torch.cat((embedded, context), dim = 2)
        #gru_input = [1, batch_size, (enc_hidden_size * 2) + embed_size]
        
        outputs, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        #output = [seq_len, batch_size, dec_hidden_size]
        #hidden = [1, batch_size, dec_hidden_size]
        
        assert (outputs == hidden).all()
        
        outputs = outputs.squeeze(0)
        context = context.squeeze(0)
        
        outputs = self.fc(torch.cat((outputs, context), dim = 1))
        #outputs = [batch_size, output_size]
        
        return outputs, hidden.squeeze(0)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_len):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_words = self.encoder.input_size
        self.max_len = max_len

    def forward(self, inputs, targets):
        #inputs = [seq_len, batch_size]
        #targets = [trg_len, batch_size]
        
        batch_size = inputs.shape[1]
        self.max_len = self.max_len if targets is None else targets.shape[0]
        
        encoder_outputs, hidden = self.encoder(inputs)
        
        decoder_outputs = torch.zeros(self.max_len, batch_size, self.n_words, device=inputs.device)
        input_token = targets[0,:] if targets is not None else torch.ones(batch_size, dtype=torch.long, device=inputs.device)
        
        for t in range(1, self.max_len):
            decoder_output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output
            
            _, top1 = decoder_output.topk(1)
            input_token = targets[t] if targets is not None else top1.squeeze(1)
            if torch.sum(input_token != 3) == 0:
                break
                
        return decoder_outputs
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def train(model, train_dataset, val_dataset, args, pt='attention.pt'):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                              num_workers=args.workers, shuffle=True, collate_fn=train_dataset.collate_fn)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    
    now = datetime.datetime.now()
    writer = SummaryWriter(f'{args.log_dir}/{pt}-{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}')
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
            
            writer.add_scalar('train_loss', loss, iters)    
        writer.add_scalar('avg_train_loss', tr_loss / (step + 1), iters)   
        val_loss = evaluate(model, val_dataset, criterion, args)
        writer.add_scalar('val_loss', val_loss, iters)
        
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

def predict(model, dataset, tokenizer, args):
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size,
                            num_workers=args.workers, collate_fn=dataset.collate_fn, shuffle=False)
    
    prediction = ''
    with torch.no_grad():
        step_iterator = tqdm(dataloader, desc='Predicting')
        for step, batch in enumerate(step_iterator):
            text = batch['text'].to(device)
            text = text.permute(1,0)
            outputs = model(text, None)
            _, outputs = outputs[1:].detach().cpu().topk(1)
            outputs = outputs.numpy().T.squeeze(0)
            for i, output in enumerate(outputs):
                pred = T.decode(output)
                pred = pred[:pred.find("</s>")]
                idx = str(int(batch['id'][0])+i)
                prediction += json.dumps({"id":idx, "predict": pred}) + '\n'

    return prediction


if __name__ == '__main__':
    
    with open('../datasets/seq2seq/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('../datasets/seq2seq/valid.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('../datasets/seq2seq/embedding.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        
    args = argparse.Namespace()
    args.n_words = len(embeddings.vocab)
    args.teacher_forcing_ratio = 0.5
    args.train_batch_size = 16
    args.val_batch_size = 16
    args.test_batch_size = 64
    args.embed_size = 300
    args.hidden_size = 300
    args.enc_dropout = 0.3
    args.dec_dropout = 0.3
    args.max_len = 80
    args.epochs = 4
    args.clip = 1
    args.lr = 1e-3
    args.workers = 8
    args.seed = 42
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn = Attention(args.hidden_size, args.hidden_size)
    enc = EncoderBiRnn(embeddings.vectors, args.hidden_size, args.enc_dropout)
    dec = DecoderRnn(embeddings.vectors, args.hidden_size, args.hidden_size, attn, args.dec_dropout)
    model = Seq2Seq(enc, dec, args.max_len).to(device)
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    model.apply(init_weights)
    
    train(model, train_data, val_data, args)