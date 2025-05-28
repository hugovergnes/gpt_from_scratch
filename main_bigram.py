"""This code simply tests out a BigramModel.
We use it to set up the scaff holding of the data and model training."""
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from dataset import create_dataloaders, make_encode_function, make_decode_function
from models.bigram_model import BigramLanguageModel


if __name__ == "__main__":
    torch.manual_seed(9021996)

    # Percentage of data that goes into the train split
    TRAINVALSPLIT = 0.8
    BATCH_SIZE = 32
    BLOCK_SIZE = 8
    NUMBER_OF_TRAIN_BATCHES = 5000
    SHOW_LOSS = True

    with open("data/input.txt", "r") as f:
        text = f.read()

    all_chars = sorted(list(set(text)))
    vocab_size = len(all_chars)
    print(f"Vocab size: {vocab_size}")

    encode = make_encode_function(all_chars)
    decode = make_decode_function(all_chars)

    # Only 65 characters.
    data = torch.tensor(encode(text), dtype=torch.uint8)
    train_data = data[: int(TRAINVALSPLIT * len(data))].clone()
    val_data = data[int(TRAINVALSPLIT * len(data)) :].clone()

    train_loader, val_loader = create_dataloaders(train_data, val_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    # print(f'Full Sentence: {train_data[:BLOCK_SIZE+1].tolist()}')
    # for batch_idx, (x, y) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"X sample (Context): {x[0].tolist()}")
    #     print(f"Y sample (Target): {y[0].tolist()}")
    #     break

    model = BigramLanguageModel(vocab_size)
    loss_function = nn.CrossEntropyLoss()

    idx = torch.zeros((1, 1)).long()

    # Very small model, thus we can use a pretty high lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx > NUMBER_OF_TRAIN_BATCHES:
            break

        logits = model(x)
        B, T, C = logits.shape
        loss = loss_function(logits.view(B*T, C), y.view(B*T))

        if SHOW_LOSS:
            losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if SHOW_LOSS:
        plt.plot(losses)
        plt.show()
    else:    
        print(loss.item())
    print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))