import argparse
import os
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from signjoey.helpers import load_config, make_logger, log_cfg
from signjoey.data_nonmap import load_data_nonmap, make_data_iter_nonmap

# ======================================================================================
# A simple model for time alignment
# ======================================================================================

class AlignmentModel(torch.nn.Module):
    """
    A simple model that takes features and projects them into an embedding space.
    For time-alignment, we'll want embeddings from different modalities 
    (e.g., face, hands, pose) for the same timestep to be close to each other.
    """
    def __init__(self, feature_size: int, embedding_size: int):
        super().__init__()
        # A simple linear layer to project features to an embedding
        self.projection = torch.nn.Linear(feature_size, embedding_size)
        # We can add more layers if needed, e.g., LSTMs to capture temporal context
        self.lstm = torch.nn.LSTM(embedding_size, embedding_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, feature_size)
        projected = self.projection(x)
        output, _ = self.lstm(projected)
        return output

# ======================================================================================
# Main Training Logic
# ======================================================================================

def train_alignment(rank: int, world_size: int, cfg_file: str):
    """Main training function for time-alignment."""
    
    cfg = load_config(cfg_file)
    model_dir = "alignment_model"
    if rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
    logger = make_logger(model_dir, mode="train", rank=rank)
    if rank == 0:
        log_cfg(cfg, logger)

    # 1. Load Data (using our simplified, keypoint-only data loader)
    if rank == 0: logger.info("Loading data...")
    train_data, _, _, _, _ = load_data_nonmap(cfg["data"])
    
    train_loader = make_data_iter_nonmap(
        dataset=train_data,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        use_ddp=(world_size > 1),
        rank=rank,
        world_size=world_size,
        num_workers=cfg["data"].get("num_workers", 4)
    )

    # 2. Build Model
    if rank == 0: logger.info("Building model...")
    # Get model dimensions from the config file
    model_cfg = cfg["model"]
    embedding_size = model_cfg.get("embedding_size", 256)
    pose_dim = model_cfg.get("alignment_pose_dim")
    hands_dim = model_cfg.get("alignment_hands_dim")
    
    if pose_dim is None or hands_dim is None:
        raise ValueError("Config must contain 'alignment_pose_dim' and 'alignment_hands_dim' under the 'model' section.")

    # We need separate models for each modality we want to align
    pose_model = AlignmentModel(pose_dim, embedding_size).to(rank)
    hand_model = AlignmentModel(hands_dim, embedding_size).to(rank)

    if world_size > 1:
        pose_model = DDP(pose_model, device_ids=[rank])
        hand_model = DDP(hand_model, device_ids=[rank])

    # 3. Optimizer and Loss
    params = list(pose_model.parameters()) + list(hand_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    loss_function = torch.nn.L1Loss() # L1 Loss to make embeddings similar

    # 4. Training Loop
    if rank == 0: logger.info("Starting training loop...")
    
    for epoch in range(100): # Run for a few epochs
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        for i, batch in enumerate(train_loader):
            if batch is None: continue

            features = batch["features"].to(rank) # (B, T, sgn_dim)
            
            # Split features into modalities using dimensions from config
            pose_features = features[:, :, :pose_dim]
            hand_features = features[:, :, pose_dim:]

            # Forward pass
            pose_embeddings = pose_model(pose_features)
            hand_embeddings = hand_model(hand_features)

            # Calculate loss
            loss = loss_function(pose_embeddings, hand_embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            if i % 10 == 0 and rank == 0:
                logger.info(f"Epoch [{epoch+1}/100], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        if rank == 0:
            logger.info("Epoch finished.")

def ddp_setup():
    """Sets up distributed data parallel."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Time-Alignment Training")
    parser.add_argument("config", type=str, help="Path to configuration file.")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        ddp_setup()
        rank = int(os.environ["RANK"])
        train_alignment(rank, world_size, args.config)
    else:
        train_alignment(0, 1, args.config) 