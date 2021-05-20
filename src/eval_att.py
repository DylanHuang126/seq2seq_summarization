from tqdm import tqdm
import numpy as np
import random
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from train_att import EncoderBiRnn, Attention, DecoderRnn, Seq2Seq
from dataset_test import Seq2SeqDataset

from torch.utils.data import Dataset
from utils import pad_to_len, Tokenizer
import pickle

parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()

with open('./datasets/seq2seq/test.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('./datasets/seq2seq/embedding.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print(len(embeddings.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attn = Attention(300, 300)
enc = EncoderBiRnn(embeddings.vectors, 300, 0.3)
dec = DecoderRnn(embeddings.vectors, 300, 300, attn, 0.3)
model = Seq2Seq(enc, dec, 80)
model.load_state_dict(torch.load('./model/attention.pt'))
model.to(device)


def predict(model, dataset, tokenizer):
    dataloader = DataLoader(dataset, batch_size=64,
                            collate_fn=dataset.collate_fn, shuffle=False)
    
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


T = Tokenizer(vocab=embeddings.vocab)
prediction = predict(model, test_data, T)

with open(args.output_path, 'w') as f:
    f.write(prediction)