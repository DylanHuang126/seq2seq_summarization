import random
import re

import spacy
import torch


class Tokenizer:
    def __init__(self, vocab=None, lower=False):
        self.nlp = spacy.load('en_core_web_sm',
                              disable=['tagger', 'ner', 'parser', 'textcat'])
        self.lower = lower
        if vocab is not None:
            self.set_vocab(vocab)

    def set_vocab(self, vocab,
                  pad_token='<pad>', bos_token='<s>',
                  eos_token='</s>', unk_token='<unk>'):
        self.vocab = vocab
        self.dict = {word: index for index, word in enumerate(self.vocab)}
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token_id = self.dict[pad_token]
        self.bos_token_id = self.dict[bos_token]
        self.eos_token_id = self.dict[eos_token]
        self.unk_token_id = self.dict[unk_token]

    def token_to_id(self, token):
        if self.lower:
            token = token.lower()
        return self.dict.get(token) or self.unk_token_id

    def collect_words(self, documents):
        words = []
        pipe = self.nlp.pipe(documents)
        if not self.lower:
            for doc in pipe:
                words += [token.text for token in doc]
        else:
            for doc in pipe:
                words += [token.text.lower() for token in doc]
        return words

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        return [token.text
                for token in self.nlp(sentence)]

    def encode(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        return [self.token_to_id(token)
                for token in self.tokenize(sentence)]

    def decode(self, indices):
        return ' '.join([self.vocab[index] for index in indices])


class Embedding:
    """
    Args:
        embedding_path (str): Path where embedding are loaded from (text file).
        words (None or list): If not None, only load embedding of the words in
            the list.
        rand_seed (int): Random seed for embedding initialization.
    """

    def __init__(self, embedding_path, words=None, rand_seed=524,
                 special_tokens=['<pad>', '<s>', '</s>', '<unk>']):
        if words is not None:
            words = set(words)

        self.vocab = special_tokens
        vectors = [[] for _ in special_tokens]
        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]
                vector = [float(v) for v in cols[1:]]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                elif word in special_tokens:
                    vectors[special_tokens.index(word)] = vector
                else:
                    self.vocab.append(word)
                    vectors.append(vector)

        random.seed(rand_seed)
        for i in range(len(special_tokens)):
            if len(vectors[i]) == 0:
                dim = len(vectors[-1])
                vectors[i] = [random.random() * 2 - 1 for _ in range(dim)]

        self.vectors = torch.tensor(vectors)


def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds
