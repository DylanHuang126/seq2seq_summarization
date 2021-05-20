import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset_test import Seq2SeqDataset
from tqdm import tqdm


def main(path):

    # loading datasets from jsonl files
    with open(path) as f:
        test = [json.loads(line) for line in f]

    with open("./datasets/seq2seq/embedding.pkl", "rb") as f:
        embedding = pickle.load(f)

    tokenizer = Tokenizer(embedding.vocab, lower=True)
    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating test dataset...')
    create_seq2seq_dataset(
        process_samples(tokenizer, test),'./datasets/seq2seq/test.pkl')


def process_samples(tokenizer, samples):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    processeds = []
    for sample in tqdm(samples):
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']) + [eos_id],
        }
        if 'summary' in sample:
            processed['summary'] = (
                [bos_id]
                + tokenizer.encode(sample['summary'])
                + [eos_id]
            )
        processeds.append(processed)

    return processeds


def create_seq2seq_dataset(samples, save_path, padding=0):
    dataset = Seq2SeqDataset(
        samples, padding=padding,
        max_text_len=300,
        max_summary_len=80
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path')
    args = parser.parse_args()
    return args.test_data_path


if __name__ == '__main__':
    path = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(path)
