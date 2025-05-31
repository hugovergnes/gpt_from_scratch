from datetime import datetime
import os
import torch

def setup_ckpt_directory(folder):
    """
    Create a checkpoint directory if it doesn't exist.

    Args:
        folder (str): Path to the checkpoint folder.

    Returns:
        str: Absolute path to the checkpoint directory.
    """
    abs_path = os.path.abspath(folder)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def save_checkpoint(model, optimizer, step, loss, checkpoint_dir="checkpoints", prefix="model"):
    """
    Save model checkpoint with timestamp in filename
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        step: Current training step
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
        prefix: Prefix for checkpoint filename
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"{prefix}_step_{step:06d}_{timestamp}.pt"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Save checkpoint with additional metadata
    checkpoint_data = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'model_config': {
            'vocab_size': getattr(model, 'token_embedding_table', torch.nn.Embedding(1,1)).num_embeddings,
            'block_size': getattr(model, 'block_size', None),
            'embed_dim': getattr(model, 'embed_dim', None),
        }
    }
    
    torch.save(checkpoint_data, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    # Also save as 'latest' for easy resuming
    latest_path = os.path.join(checkpoint_dir, f"{prefix}_latest.pt")
    torch.save(checkpoint_data, latest_path)
    print(f"Latest checkpoint saved: {latest_path}")
    
    return filepath