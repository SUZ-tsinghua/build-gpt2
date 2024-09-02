from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt2_model import GPT2, GPT2Config
from gpt2_dataloader import DataLoaderLite
import math
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# -------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for 4 GPTs:
# torchrun --standalone --nproc_per_node=4 train_gpt2.py

# set up DDP (distributed data parallel)
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"Running on {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

# set random seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation
total_batch_size = 524288 # 2^19, in num of tokens
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> gradient accumulation steps: {grad_accum_steps}")

# create the data loader
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision("high")

# create the model
model = GPT2(GPT2Config(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    print(f"creating DDP model {ddp_local_rank}")
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) 
    if step > max_steps:
        return min_lr
    # 3) cosine annealing from max_lr to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(decay_ratio * math.pi))
    return min_lr + (max_lr - min_lr) * coeff

# optimize!
import time
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip the gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # learning rate schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    if master_process:
        print(f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f} | dt {dt:.2f}ms")

if ddp:
    destroy_process_group()

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

