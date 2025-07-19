"""Regroup Dataset and Dataloader into the same file.
Usually they are splitted in 2.
"""
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

def make_encode_function(all_chars):
    """Encode character to integer

    Args:
        c (_type_): _description_
    """
    char_to_int = {c: idx for idx, c in enumerate(all_chars)}
    return lambda x: [char_to_int[c] for c in x]


def make_decode_function(all_chars):
    int_to_char = {idx: c for idx, c in enumerate(all_chars)}
    return lambda x: ''.join([int_to_char[c] for c in x])

class TokenDataset(Dataset):
    def __init__(self, npy_file, block_size):
        """
        Dataset for loading pre-tokenized OpenWebText stored as a .npy array.
        
        Args:
            npy_file (str or Path): Path to .npy file with shape (N, block_size)
            block_size (int): Sequence length
        """
        tokens = np.load(npy_file)
        if tokens.ndim == 2:
            self.data = torch.tensor(tokens.flatten(), dtype=torch.long)
        elif tokens.ndim == 1:
            self.data = torch.tensor(tokens, dtype=torch.long)
        else:
            raise ValueError("Unsupported .npy shape: expected 1D or 2D array.")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class CharacterDataset(Dataset):
    def __init__(self, data, block_size):
        """
        Args:
            data: torch.tensor of character indices
            block_size: context length (number of characters to use as input)
        """
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        # We can create len(data) - block_size samples
        # Each sample uses block_size chars as X and 1 char as Y
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # X: context of block_size characters starting at idx
        # Y: the next character (target to predict)
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

def create_dataloaders(
    train_data,
    val_data,
    batch_size=32,
    block_size=8,
    evals_iter=None,
    dataset_type="char"
):
    """
    Create train and validation dataloaders
    
    dataset_type: 'char' or 'token'
    """
    if dataset_type == "char":
        train_dataset = CharacterDataset(train_data, block_size)
        val_dataset = CharacterDataset(val_data, block_size)
    elif dataset_type == "token":
        train_dataset = TokenDataset(train_data, block_size)
        val_dataset = TokenDataset(val_data, block_size)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    if evals_iter is not None:
        val_sampler = RandomSampler(val_dataset, num_samples=evals_iter)
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler
    )

    return train_loader, val_loader

if __name__ == '__main__':
    torch.manual_seed(9021996)

    # Percentage of data that goes into the train split
    TRAINVALSPLIT = 0.8
    BATCH_SIZE = 1
    BLOCK_SIZE = 8

    with open("data/input.txt", "r") as f:
        text = f.read()

    all_chars = sorted(list(set(text)))
    print(f"Vocab size: {len(all_chars)}")

    encode = make_encode_function(all_chars)
    decode = make_decode_function(all_chars)

    # Only 65 characters.
    data = torch.tensor(encode(text), dtype=torch.uint8)
    train_data = torch.tensor(data[: int(TRAINVALSPLIT * len(data))])
    val_data = torch.tensor(data[int(TRAINVALSPLIT * len(data)) :])

    train_loader, val_loader = create_dataloaders(train_data, val_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    print(f'Full Sentence: {train_data[:BLOCK_SIZE+1]}')
    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"X sample (Context): {x[0]}")
        print(f"Y sample (Target): {y[0]}")
        break