import torch

## Data Preprocessing
data_file = 'data/full.zip'
train_file = 'data/train.gz'
val_file = 'data/val.gz'

train_tkn_file = 'data/train.tkn.gz'
val_tkn_file = 'data/val.tkn.gz'

train_contexts_file = 'data/train.contexts.gz'
val_contexts_file = 'data/val.contexts.gz'

vocab_file = 'data/vocab.pkl'
special_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']
unk_token = '[UNK]'
pad_idx = 0
max_words = 10000
min_freq = 5
context_size = 10

## Training
batch_size = 1024
lr = 1e-3
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Model
embedding_dim = 100
hidden_dim = 256
model_file = 'models/model.pt'
ckpt_file = 'models/ckpt.pt'
