# import math
# import inspect
# from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Simplest possible LLM.

    It predicts the next character (or token) based only on the immediately
    preceding character.

    """
    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table for logits. Init with a random value.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    
    def forward(self, idx):
        """
        B: Batch size
        T: Block size
        C: Number of channels after embedding. (vocab size)

        Args:
            idx (torch.Tensor): (B, T) tensor 
            targets (torch.Tensor): (B, T) tensor 
        
        Returns:
            A tensor of size B, T, C wher
        """
        # Embedding in pytorch expects a long for memory adressing.
        logits = self.token_embedding_table(idx.long()) 
        return logits
    
    # Generate for a generic Model
    # def generate(self, idx, max_new_tokens):
    #     """Continues the generation in the time dimension.

    #     Args:
    #         idx (torch.Tensor): Input tensor
    #         max_new_tokens (int): Number of generated tokens.

    #     Returns:
    #         torch.Tensor: Larger tensor of size .....
    #     """
    #     for _ in range(max_new_tokens):
    #         logits = self(idx)
    #         # Look at the last logit, this is the prediction for what comes next
    #         logits = logits[:, -1, :]
    #         # Now sample from the model distribution to generate a new token.
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx
    
    def generate(self, idx, max_new_tokens):
        """Continues the generation in the time dimension.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T)
            max_new_tokens (int): Number of generated tokens.

        Returns:
            torch.Tensor: Larger tensor of size .....
        """

        for _ in range(max_new_tokens):
            # Bigram model only needs the previous character
            logits = self(idx[:, -1])
            # Now sample from the model distribution to generate a new token.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

        