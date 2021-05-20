#!/usr/bin/env bash

wget https://www.dropbox.com/s/gm9t698ckm8bgwh/embedding.pkl?dl=1 -O ./datasets/seq2seq/embedding.pkl
wget https://www.dropbox.com/s/cm6klztud81abpj/embedding_seq_tag.pkl?dl=1 -O ./datasets/seq_tag/embedding.pkl

#extractive
wget https://www.dropbox.com/s/s4a0hxg8mnk8khi/extmodel.pt?dl=1 -O ./model/extractive.pt
#abstractive
wget https://www.dropbox.com/s/i8wqor55jpa11qc/s2smodel.pt?dl=1 -O ./model/abstractive.pt
#attention
wget https://www.dropbox.com/s/9deia65zeub2cwk/attmodel.pt?dl=1 -O ./model/attention.pt

pip install -r requirements.txt
python -m nltk.downloader all
python -m spacy download en_core_web_sm