import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional

class MediaPipeHandExtractor:
    def __init__(self):
        """
        Initializing the MediaPipe Hand Feature Extractor
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_features(self, video_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Extract hand features from a video
        
        Args:
            video_path: Input video path
            output_path: Feature save path (optional)
            
        Returns:
            np.ndarray: Extracted hand features (N x F)
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract hands using MediaPipe
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                # Convert hand landmarks to features
                hand_features = self._process_landmarks(results.multi_hand_landmarks)
                features.append(hand_features)
            else:
                # If no hands are detected, fill with zeros
                features.append(np.zeros(21 * 2 * 2))  # 84
        
        cap.release()
        
        # Convert features to numpy array
        features = np.array(features)
        
        # Save if output path is specified
        if output_path:
            np.save(output_path, features)
            
        return features
    
    def _process_landmarks(self, landmarks_list: List[Any]) -> np.ndarray:
        """
        Convert hand landmarks to feature vectors
        
        Args:
            landmarks_list: MediaPipe hand landmarks list
            
        Returns:
            np.ndarray: Processed feature vector (84,)
        """
        all_hand_coords = []
        for hand_landmarks in landmarks_list:
            for lm in hand_landmarks.landmark:
                all_hand_coords.extend([lm.x, lm.y])

        # Add padding based on the number of detected hands to fix the size to 84
        num_coords = len(all_hand_coords)
        padding_needed = (21 * 2 * 2) - num_coords
        if padding_needed > 0:
            all_hand_coords.extend([0] * padding_needed)
            
        return np.array(all_hand_coords[:84]) 