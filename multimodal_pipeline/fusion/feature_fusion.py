import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class FeatureFusion(nn.Module):
    def __init__(self, pose_dim: int, hand_dim: int, mouth_dim: int, hidden_dim: int = 512):
        """
        Initializing the Multimodal Feature Fusion Module
        
        Args:
            pose_dim: Pose feature dimension
            hand_dim: Hand feature dimension
            mouth_dim: Mouth feature dimension
            hidden_dim: Hidden dimension of the Transformer encoder
        """
        super().__init__()
        
        # Concatenate each feature and project to the input dimension of the Transformer
        self.input_dim = pose_dim + hand_dim + mouth_dim
        self.projection = nn.Linear(self.input_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # Set input order to (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, pose_features: torch.Tensor, hand_features: torch.Tensor, 
                mouth_features: torch.Tensor) -> torch.Tensor:
        """
        Multimodal feature fusion
        
        Args:
            pose_features: Pose features (B x T x pose_dim)
            hand_features: Hand features (B x T x hand_dim)
            mouth_features: Mouth features (B x T x mouth_dim)
            
        Returns:
            torch.Tensor: Fused features (B x T x hidden_dim)
        """
        # Combine features along the last dimension (feature dimension)
        combined = torch.cat([pose_features, hand_features, mouth_features], dim=2)
        
        # Project to the input dimension of the Transformer
        projected = self.projection(combined)
        
        # Pass through the Transformer encoder
        fused = self.transformer(projected)
        
        return fused

def fuse_features(pose_features: np.ndarray, hand_features: np.ndarray, 
                 mouth_features: np.ndarray, model: Optional[FeatureFusion] = None) -> np.ndarray:
    """
    Execute feature fusion
    
    Args:
        pose_features: Pose features (N x pose_dim)
        hand_features: Hand features (N x hand_dim)
        mouth_features: Mouth features (N x mouth_dim)
        model: Fusion model (optional)
        
    Returns:
        np.ndarray: Fused features
    """
    # Convert numpy arrays to torch tensors
    pose_tensor = torch.from_numpy(pose_features).float().unsqueeze(0) # Add batch dimension
    hand_tensor = torch.from_numpy(hand_features).float().unsqueeze(0) # Add batch dimension
    mouth_tensor = torch.from_numpy(mouth_features).float().unsqueeze(0) # Add batch dimension
    
    if model is None:
        # If no model is provided, simply concatenate the features
        fused = np.concatenate([pose_features, hand_features, mouth_features], axis=1)
    else:
        # Fusion through the model
        model.eval()
        with torch.no_grad():
            fused_tensor = model(pose_tensor, hand_tensor, mouth_tensor)
            fused = fused_tensor.squeeze(0).numpy() # Remove batch dimension
            
    return fused 