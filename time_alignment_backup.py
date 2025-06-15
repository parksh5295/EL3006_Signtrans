# time_alignment.py
# Time Alignment Module for aligning pose, hand, and mouth features.
# Framework: PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # (B, D, T) -> (B, T, D)
        return x.transpose(1, 2)

class SoftTimeAlign(nn.Module):
    def __init__(self, input_dim, target_len):
        super().__init__()
        self.target_len = target_len
        self.align_proj = nn.Linear(input_dim, target_len)

    def forward(self, x):
        # x: (B, T, D)
        attn_weights = self.align_proj(x)  # (B, T, target_len)
        attn_weights = F.softmax(attn_weights, dim=1)  # soft temporal alignment
        x_aligned = torch.einsum("btd,btn->bnd", x, attn_weights)  # (B, target_len, D)
        return x_aligned

class TimeAlignmentModule(nn.Module):
    def __init__(self, pose_dim, hand_dim, mouth_dim, hidden_dim, target_len):
        super().__init__()
        self.pose_encoder = TemporalEncoder(pose_dim, hidden_dim)
        self.hand_encoder = TemporalEncoder(hand_dim, hidden_dim)
        self.mouth_encoder = TemporalEncoder(mouth_dim, hidden_dim)

        self.pose_align = SoftTimeAlign(hidden_dim, target_len)
        self.hand_align = SoftTimeAlign(hidden_dim, target_len)
        self.mouth_align = SoftTimeAlign(hidden_dim, target_len)

    def forward(self, pose, hand, mouth):
        # pose, hand, mouth: (B, T_x, D_x)
        pose_feat = self.pose_encoder(pose)
        hand_feat = self.hand_encoder(hand)
        mouth_feat = self.mouth_encoder(mouth)

        aligned_pose = self.pose_align(pose_feat)
        aligned_hand = self.hand_align(hand_feat)
        aligned_mouth = self.mouth_align(mouth_feat)

        return aligned_pose, aligned_hand, aligned_mouth

# Example usage:
# model = TimeAlignmentModule(pose_dim=34, hand_dim=42, mouth_dim=24, hidden_dim=64, target_len=75)
# pose_input = torch.randn(1, 100, 34) # Example pose input with 100 frames
# hand_input = torch.randn(1, 120, 42) # Example hand input with 120 frames
# mouth_input = torch.randn(1, 90, 24) # Example mouth input with 90 frames
# aligned_pose, aligned_hand, aligned_mouth = model(pose_input, hand_input, mouth_input)
# print(aligned_pose.shape) # torch.Size([1, 75, 64])
# print(aligned_hand.shape) # torch.Size([1, 75, 64])
# print(aligned_mouth.shape) # torch.Size([1, 75, 64]) 