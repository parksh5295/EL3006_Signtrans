import os
from typing import Optional, Dict, Any
import numpy as np
import torch

from .feature_extractors.openpose_extractor import OpenPoseExtractor
from .feature_extractors.lipnet_extractor import LipNetExtractor
from .feature_extractors.mediapipe_extractor import MediaPipeHandExtractor
from .fusion.feature_fusion import FeatureFusion, fuse_features
from .utils.video_utils import get_video_info, resize_video

class MultimodalPipeline:
    def __init__(self, model_paths: Dict[str, str], output_dir: str):
        """
        Initializing the Multimodal Pipeline
        
        Args:
            model_paths: Dictionary containing paths to each model
                {
                    'openpose': 'path/to/openpose/model',
                    'lipnet': 'path/to/lipnet/model'
                }
            output_dir: Output directory path
        """
        self.model_paths = model_paths
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initializing feature extractors
        self.pose_extractor = OpenPoseExtractor(model_paths['openpose'])
        self.lipnet_extractor = LipNetExtractor(model_paths['lipnet'])
        self.hand_extractor = MediaPipeHandExtractor()
        
        # Initializing feature fusion model
        self.fusion_model = FeatureFusion(
            pose_dim=75,  # 25 keypoints * 3 coordinates
            hand_dim=84,  # 2 hands * 21 landmarks * 2 coordinates
            mouth_dim=100,  # LipNet output dimension
            hidden_dim=512
        )
        
    def process_video(self, video_path: str, output_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Processing a video and extracting features
        
        Args:
            video_path: Input video path
            output_name: Output file name (optional)
            
        Returns:
            Dict[str, np.ndarray]: Extracted feature dictionary
        """
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(video_path))[0]
            
        # Checking video information
        width, height, total_frames, fps = get_video_info(video_path)
        
        # Extracting features
        pose_features = self.pose_extractor.extract_features(
            video_path,
            os.path.join(self.output_dir, f"{output_name}_pose.npy")
        )
        
        hand_features = self.hand_extractor.extract_features(
            video_path,
            os.path.join(self.output_dir, f"{output_name}_hand.npy")
        )
        
        mouth_features = self.lipnet_extractor.extract_features(
            video_path,
            os.path.join(self.output_dir, f"{output_name}_mouth.npy")
        )
        
        # Fusing features
        fused_features = fuse_features(
            pose_features,
            hand_features,
            mouth_features,
            self.fusion_model
        )
        
        # Saving fused features
        np.save(
            os.path.join(self.output_dir, f"{output_name}_fused.npy"),
            fused_features
        )
        
        return {
            'pose': pose_features,
            'hand': hand_features,
            'mouth': mouth_features,
            'fused': fused_features
        } 