import numpy as np
import torch
import os

# Import the test target module
from multimodal_pipeline.fusion.feature_fusion import FeatureFusion, fuse_features

def test_feature_fusion_pipeline():
    """
    Test function to verify the integration logic of the feature fusion pipeline.
    Instead of running actual models, we only check the fusion process with dummy data.
    """
    print("Start testing feature fusion pipeline...")

    # 1. Create dummy data for testing
    print("1. Create dummy feature data...")
    num_frames = 50
    pose_dim = 75
    hand_dim = 84
    mouth_dim = 100
    hidden_dim = 512

    dummy_pose_features = np.random.rand(num_frames, pose_dim).astype(np.float32)
    dummy_hand_features = np.random.rand(num_frames, hand_dim).astype(np.float32)
    dummy_mouth_features = np.random.rand(num_frames, mouth_dim).astype(np.float32)

    print(f"   - Pose feature size: {dummy_pose_features.shape}")
    print(f"   - Hand feature size: {dummy_hand_features.shape}")
    print(f"   - Mouth feature size: {dummy_mouth_features.shape}")

    # 2. Initialize the fusion model
    print("\n2. Initialize the FeatureFusion model...")
    fusion_model = FeatureFusion(
        pose_dim=pose_dim,
        hand_dim=hand_dim,
        mouth_dim=mouth_dim,
        hidden_dim=hidden_dim
    )
    print(f"   - Model initialized. Hidden dimension: {hidden_dim}")

    # 3. Execute feature fusion
    print("\n3. Execute feature fusion using `fuse_features` function...")
    fused_features = fuse_features(
        dummy_pose_features,
        dummy_hand_features,
        dummy_mouth_features,
        model=fusion_model
    )
    print("   - Feature fusion completed.")

    # 4. Verify the result
    print("\n4. Verify the shape of the fused features...")
    expected_shape = (num_frames, hidden_dim)
    actual_shape = fused_features.shape
    
    print(f"   - Expected shape: {expected_shape}")
    print(f"   - Actual shape: {actual_shape}")

    assert actual_shape == expected_shape, f"Test failed: The final feature shape does not match the expected shape!"
    
    print("\nTest passed! The integration logic is working correctly.")

if __name__ == "__main__":
    test_feature_fusion_pipeline() 