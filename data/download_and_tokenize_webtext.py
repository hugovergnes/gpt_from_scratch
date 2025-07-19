import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import argparse

def tokenize_and_chunk(text, tokenizer, block_size):
    """
    Tokenizes a text string and splits it into non-overlapping fixed-length chunks.

    Args:
        text (str): The raw text to tokenize.
        tokenizer: The tokenizer object used to convert text to tokens.
        block_size (int): The length of each output chunk.

    Returns:
        List[List[int]]: A list of token chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i + block_size]
        for i in range(0, len(tokens) - block_size + 1, block_size)
    ]
    return chunks

def save_tokenized_chunks(chunks, out_file):
    """
    Saves tokenized chunks to a .npy file, appending if the file already exists.

    Args:
        chunks (List[List[int]]): Tokenized sequences to save.
        out_file (str): Path to the output .npy file.
    """
    if os.path.exists(out_file):
        existing = np.load(out_file)
        all_chunks = np.concatenate([existing, np.array(chunks, dtype=np.uint16)])
    else:
        all_chunks = np.array(chunks, dtype=np.uint16)
    np.save(out_file, all_chunks)

def main(out_dir="tokenized_openwebtext", block_size=128, split="train", max_docs=None, train_split=0.9):
    """
    Tokenizes OpenWebText dataset into fixed-length chunks using GPT-2 tokenizer and saves them as .npy files.

    Args:
        out_dir (str, optional): Directory to save the tokenized dataset. Defaults to "tokenized_openwebtext".
        block_size (int, optional): Length of each token chunk. Defaults to 128.
        split (str, optional): Split of the OpenWebText dataset to load. Defaults to "train".
        max_docs (int, optional): Limit the number of documents to tokenize. Defaults to None (all documents).
        train_split (float, optional): Ratio of data to use for training (rest goes to validation). Defaults to 0.9.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"openwebtext_tokens_{block_size}.npy")

    print("🔄 Loading OpenWebText...")
    dataset = load_dataset("openwebtext", split=split)
    if max_docs:
        dataset = dataset.select(range(max_docs))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    all_chunks = []
    for example in tqdm(dataset, desc="🧠 Tokenizing"):
        chunks = tokenize_and_chunk(example["text"], tokenizer, block_size)
        all_chunks.extend(chunks)

    print(f"💾 Saving {len(all_chunks)} chunks of {block_size} tokens...")

    # Split into train/val and save separately
    split_idx = int(train_split * len(all_chunks))
    train_chunks = np.array(all_chunks[:split_idx], dtype=np.uint16)
    val_chunks = np.array(all_chunks[split_idx:], dtype=np.uint16)

    train_file = os.path.join(out_dir, f"openwebtext_tokens_{block_size}_train.npy")
    val_file = os.path.join(out_dir, f"openwebtext_tokens_{block_size}_val.npy")

    print(f"💾 Saving {len(train_chunks)} train chunks to {train_file}")
    np.save(train_file, train_chunks)

    print(f"💾 Saving {len(val_chunks)} val chunks to {val_file}")
    np.save(val_file, val_chunks)
    print(f"✅ Done! Saved to {train_file} and {val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="tokenized_openwebtext")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_docs", type=int, default=None, help="Truncate to first N documents for testing")
    args = parser.parse_args()

    main(out_dir=args.out_dir, block_size=args.block_size, split=args.split, max_docs=args.max_docs)

    # python download_and_tokenize_openwebtext.py --max_docs 1000
    # python download_and_tokenize_openwebtext.py