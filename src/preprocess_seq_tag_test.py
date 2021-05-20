import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Iterable

from dataset_test import SeqTaggingDataset
from tqdm import tqdm
from utils import Tokenizer, Embedding


def main(path):
    with open(path) as f:
        test = [json.loads(line) for line in f]


    with open("./datasets/seq_tag/embedding.pkl", "rb") as f:
        embedding = pickle.load(f)

    tokenizer = Tokenizer(embedding.vocab, lower=True)
    tokenizer.set_vocab(embedding.vocab)

    logging.info('Creating test dataset...')
    create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, test),
        './datasets/seq_tag/test.pkl'
    )


def process_seq_tag_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        if not sample['sent_bounds']:
            continue
        processed = {
            'id': sample['id'],
            'text': tokenizer.encode(sample['text']),
            'sent_range': get_tokens_range(tokenizer, sample)
        }

        if 'extractive_summary' in sample:
            label_start, label_end = processed['sent_range'][sample['extractive_summary']]
            processed['label'] = [
                1 if label_start <= i < label_end else 0
                for i in range(len(processed['text']))
            ]
            assert len(processed['label']) == len(processed['text'])
        processeds.append(processed)
    return processeds


def get_tokens_range(tokenizer,
                     sample) -> Iterable:
    ranges = []
    token_start = 0
    for char_start, char_end in sample['sent_bounds']:
        sent = sample['text'][char_start:char_end]
        tokens_in_sent = tokenizer.tokenize(sent)
        token_end = token_start + len(tokens_in_sent)
        ranges.append((token_start, token_end))
        token_start = token_end
    return ranges


def create_seq_tag_dataset(samples, save_path, padding=0):
    dataset = SeqTaggingDataset(
        samples, padding=padding,
        max_text_len=300
    )
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path')
    args = parser.parse_args()
    return args.test_data_path


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
