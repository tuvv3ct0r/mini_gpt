import os
import requests
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


from config.gpt_config import GPTConfig

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

char_to_id = {ch:i for i, ch in enumerate(chars)}
id_to_char = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [char_to_id[c] for c in s]

def decode(l):
    return ''.join([id_to_char[i] for i in l])

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': id_to_char,
    'stoi': char_to_id,
}
with open(os.path.join(os.path.dirname(__file__),  'data/meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

config = GPTConfig()
batch_size = config.batch_size
block_size = config.max_seq_len

def get_batch(mode):
    if mode == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.from_numpy(np.stack([data[i:i+block_size] for i in ix])).long()
    y = torch.from_numpy(np.stack([data[i+1:i+block_size+1] for i in ix])).long()
    return x, y

# --- Training Loop ---
from model.gpt import GPT

config = GPTConfig()
# Load meta.pkl to set vocab_size
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
config.vocab_size = meta['vocab_size']

device = config.device
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {num_params:,}")

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(config.eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

for iter in trange(5, desc="Training"):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iter {iter}: loss {loss.item():.4f}")

    if iter % 50 == 0:
        # Sample from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample = model.generate(context, max_new_tokens=100)
        decoded = decode(sample[0].tolist())
        print(f"Sample at iter {iter}:\n{decoded}\n{'-'*40}")

    if iter % 100 == 0:
        # Save model checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), f"model_ckpt_{iter}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint at {ckpt_path}")

    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")