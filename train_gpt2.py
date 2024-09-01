from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt2_model import GPT2, GPT2Config
from gpt2_dataloader import DataLoaderLite

# -------------------------------------------------------------
# create the data loader
train_loader = DataLoaderLite(4, 32)

# create the model
model = GPT2(GPT2Config)
model.to("cuda")
model.train()

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss {loss.item()}")

import sys; sys.exit()

# tokenize prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to("cuda")

# generate
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        # get the probabilities for the next token
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1) # (num_return_sequences, vocab_size)
        
        # do top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (num_return_sequences, 50)

        # sample from the top-k
        ix = torch.multinomial(topk_probs, 1) # (num_return_sequences, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (num_return_sequences, 1)

        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated sequences
for i in range(num_return_sequences):
    decoded = enc.decode(x[i, :max_length].tolist())
    print(">", decoded)

