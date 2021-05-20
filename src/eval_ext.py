from tqdm import tqdm
import numpy as np
import random
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from train_ext import LSTM
from dataset_test import SeqTaggingDataset

from torch.utils.data import Dataset
import pickle

parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()

with open('./datasets/seq_tag/test.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('./datasets/seq_tag/embedding.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print(len(embeddings.vocab))
    
device = torch.device('cuda')
model = LSTM(input_size=300, hidden_size=128, n_layers=2, n_cls=1, embedding_weight=embeddings.vectors).to(device)
model.load_state_dict(torch.load('./model/extractive.pt'))
model.to(device)
model.eval()

def predict_topk(model, val_data, top_k=2):
    val_loader = DataLoader(val_data, batch_size=1, 
                            collate_fn=val_data.collate_fn)
    all_ext = []
    all_n_sent = []
    prediction = ''
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            x = batch['text'].to(device)

            if len(x[0]) == 0:
                prediction += json.dumps({"id":batch['id'][0], "predict_sentence_index": [0]}) + '\n'
                continue
            pred = torch.sigmoid(model(x)).squeeze(-1)
            pred = pred[0] > 0.5
            ext = []
            range_tup = batch['sent_range'][0]
            for tup in range_tup:
                ext.append(float(sum(pred[tup[0]:tup[1]])) / (tup[1]-tup[0]+1))
            
            topk = min(top_k, len(ext))
            ans = np.array(ext).argsort()[-topk:][::-1].tolist()
            
            if len(ext) == 0:
                ext.append(0)
            else:
                ext = ans
            ext.sort()
            
            n_sent = len(batch['sent_range'][0])
            all_n_sent.append(n_sent)
            all_ext.append(ext)
            prediction += json.dumps({"id":batch['id'][0], "predict_sentence_index": ext}) + '\n'
    
    return prediction, all_ext, all_n_sent

prediction, all_ext , all_n_sent = predict_topk(model, test_data)

with open(args.output_path, 'w') as f:
    f.write(prediction)