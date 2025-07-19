"""This code simply train the Transformer model."""
from datetime import datetime
import torch
import torch.nn as nn
import tiktoken
import time
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from models import GPT
from utils.checkpoint import setup_ckpt_directory, save_checkpoint


def run_evaluation(model, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device=device), y_val.to(device=device)
            logits = model(x_val)
            B, T, C = logits.shape
            val_loss = loss_function(logits.view(B*T, C), y_val.view(B*T))
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f'[Eval] Step {batch_idx}: Val Loss = {avg_val_loss:.4f} [{current_time}]')
    return avg_val_loss

def set_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def set_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # TODO: Check torch is >2.1
    elif torch.mps.is_available():
        return torch.bfloat16
    else:
        return torch.float32

if __name__ == "__main__":
    # Seed
    torch.manual_seed(1337)

    # Prepare folders for checkpoints and data.
    setup_ckpt_directory('checkpoints')

    # Config
    # Percentage of data that goes into the train split
    TRAINVALSPLIT = 0.9
    BATCH_SIZE = 12
    BLOCK_SIZE = 64
    NUMBER_OF_TRAIN_BATCHES = 2000
    SHOW_LOSS = False
    learning_rate = 1e-3
    number_of_layers = 8
    number_of_heads = 8
    embed_dim = 256
    dropout=0.2
    n_evals = 100
    log_every = 50 # Log the throughput every x batches.
    evals_iter = 500 # Evals every n iters
    save_interval = 2000  # Save checkpoint every n steps

    compile = False
    
    device = set_device()
    # Float 16 will not work on CPU.
    # Bloat16 is experimental on mps device.
    dtype = set_dtype()


    print(f'Device: {device}')
    print(f'Dtype: {dtype}')

    with open("data/input.txt", "r") as f:
        text = f.read()
    
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    vocab_size = enc.n_vocab

    print(f"Vocab size: {vocab_size}")

    data = torch.tensor(encode(text), dtype=torch.long)
    train_data = data[: int(TRAINVALSPLIT * len(data))].clone()
    val_data = data[int(TRAINVALSPLIT * len(data)) :].clone()

    train_loader, val_loader = create_dataloaders(train_data, val_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, evals_iter=evals_iter)

    model = GPT(vocab_size,
                BLOCK_SIZE,
                number_of_layers=number_of_layers,
                number_of_heads=number_of_heads,
                embed_dim=embed_dim,
                dropout=dropout)
    model = model.to(device=device, dtype=dtype)
    if compile:
        model = torch.compile(model)

    loss_function = nn.CrossEntropyLoss()

    # Very small model, thus we can use a pretty high lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)


    losses = []
    print('Starting training...')
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx > NUMBER_OF_TRAIN_BATCHES:
            break
        
        start = time.perf_counter()
        x, y = x.to(device=device), y.to(device=device)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast():
                logits = model(x)
                loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            logits = model(x)
            loss = loss_function(logits.view(-1, logits.size(-1)), y.view(-1))


        scaler.scale(loss).backward() if use_amp else loss.backward()
        scaler.step(optimizer) if use_amp else optimizer.step()
        if use_amp:
            scaler.update()
        end = time.perf_counter()
        if batch_idx % log_every == 0:
            elapsed = end - start
            samples_per_sec = x.shape[0] / elapsed
            print(f"[Step {batch_idx}] Throughput: {samples_per_sec:.2f} samples/sec")
        
        # Save
        if batch_idx > 0 and batch_idx % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=batch_idx,
                loss=loss.item(),
                prefix="gpt"
            )

        # Eval
        if batch_idx % n_evals == 0:
            avg_val_loss = run_evaluation(model, device)
            if SHOW_LOSS:
                losses.append(avg_val_loss)
            model.train()
    
    if SHOW_LOSS:
        plt.plot(losses)
        plt.show()
    else:    
        print(loss.item())

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(loss.item())
    print(decode(model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=200)[0].tolist()))