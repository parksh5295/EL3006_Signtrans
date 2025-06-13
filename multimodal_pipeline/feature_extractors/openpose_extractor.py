import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional

class OpenPoseExtractor:
    def __init__(self, model_path: str):
        """
        Initializing the OpenPose Feature Extractor
        
        Args:
            model_path: OpenPose model path
        """
        self.model_path = model_path
        # Initializing OpenPose Python API
        try:
            import openpose
            self.op = openpose.OpenPose()
        except ImportError:
            raise ImportError("OpenPose Python API is not installed.")

    def extract_features(self, video_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Extract pose features from a video
        
        Args:
            video_path: Input video path
            output_path: Feature save path (optional)
            
        Returns:
            np.ndarray: Extracted pose features (N x F)
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract pose using OpenPose
            datum = self.op.forward(frame)
            if datum.poseKeypoints is not None and len(datum.poseKeypoints.shape) > 0:
                # Convert pose keypoints to features
                pose_features = self._process_keypoints(datum.poseKeypoints)
                features.append(pose_features)
            else:
                # If no person is detected, fill with zeros
                features.append(np.zeros(25 * 3))
        
        cap.release()
        
        # Convert features to numpy array
        features = np.array(features)
        
        # Save if output path is specified
        if output_path:
            np.save(output_path, features)
            
        return features
    
    def _process_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Convert keypoints to feature vectors (only process the most confident person)
        
        Args:
            keypoints: OpenPose keypoints (N_people x 25 x 3)
            
        Returns:
            np.ndarray: Processed feature vector (75,)
        """
        # Use only the keypoints of the most confident person (first person)
        if keypoints.ndim > 2 and keypoints.shape[0] > 0:
            return keypoints[0].flatten()
        elif keypoints.ndim == 2:  # If only one person is detected
            return keypoints.flatten()
        else: # If no person is detected
            return np.zeros(25 * 3) 