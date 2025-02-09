import os, gzip
import pandas as pd
import spacy

import config
from vocab import Vocab

def split_data(file, train_file, val_file, ratio):
    if os.path.exists(train_file) or os.path.exists(val_file):
        print('Found train/val files, skipping split...')
        return

    print("Splitting data...")

    df = pd.read_csv(file)

    ## shrink data for small model testing
    df = df.sample(frac=0.25, random_state=1024)

    train_df = df.sample(frac=ratio, random_state=1024)
    val_df = df.drop(train_df.index)

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print("Data split saved to", train_file, val_file)

def create_vocab(file, vocab_file):
    vocab = Vocab(max_words=config.max_words, min_freq=config.min_freq,
                  special_tokens=config.special_tokens, default_token=config.unk_token)
    vocab.build_vocab_from_file(file, vocab_file)
    return vocab

def tokenize_data(file, vocab, out_file):
    if os.path.exists(out_file):
        print(f'Found {out_file}, skipping tokenization...')
        return

    print("Tokenizing data...")

    nlp = spacy.blank('fr')
    df = pd.read_csv(file)

    with gzip.open(out_file, 'wt') as f:
        f.write('text\n')
        for text in df['text']:
            tokens = [token.text for token in nlp(text.lower()) if not token.is_punct]
            tokens = [vocab[token] for token in tokens]
            tokens = ' '.join(map(str, tokens))
            f.write(f'{tokens}\n')

    print("Tokenized data saved to", out_file)

def create_contexts(f_in, f_out, n):
    if os.path.exists(f_out):
        print(f'Found {f_out}, skipping context creation...')
        return

    print(f"Creating contexts...")

    with gzip.open(f_in, 'rt') as f, gzip.open(f_out, 'wt') as out:
        out.write('text\n')
        next(f)
        for line in f:
            tokens = line.strip().split()
            context = [tokens[i:i+n] for i in range(len(tokens)-n+1)]
            context = [' '.join(con) for con in context]
            context = '\n'.join(context)
            out.write(f'{context}\n')

    print(f"Contexts saved to", f_out)

if __name__ == '__main__':
    split_data(config.data_file, config.train_file, config.val_file, 0.8)
    vocab = create_vocab(config.train_file, config.vocab_file)

    tokenize_data(config.train_file, vocab, config.train_tkn_file)
    tokenize_data(config.val_file, vocab, config.val_tkn_file)

    create_contexts(config.train_tkn_file, config.train_contexts_file, config.context_size)
    create_contexts(config.val_tkn_file, config.val_contexts_file, config.context_size)
