import cv2
import numpy as np
from typing import List, Tuple, Optional

def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """
    Get basic information about a video file
    
    Args:
        video_path: Video file path
        
    Returns:
        Tuple[int, int, int, float]: (width, height, total_frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return width, height, total_frames, fps

def extract_frames(video_path: str, start_frame: Optional[int] = None, 
                  end_frame: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video
    
    Args:
        video_path: Video file path
        start_frame: Start frame number (optional)
        end_frame: End frame number (optional)
        
    Returns:
        List[np.ndarray]: Extracted frame list
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if start_frame is not None and frame_idx < start_frame:
            frame_idx += 1
            continue
            
        if end_frame is not None and frame_idx >= end_frame:
            break
            
        frames.append(frame)
        frame_idx += 1
        
    cap.release()
    return frames

def save_video(frames: List[np.ndarray], output_path: str, fps: float = 30.0):
    """
    Save a list of frames to a video file
    
    Args:
        frames: List of frames
        output_path: Output video file path
        fps: Frames per second
    """
    if not frames:
        return
        
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
        
    out.release()

def resize_video(video_path: str, output_path: str, target_size: Tuple[int, int]):
    """
    Resize a video
    
    Args:
        video_path: Input video file path
        output_path: Output video file path
        target_size: (width, height) tuple
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        resized = cv2.resize(frame, target_size)
        out.write(resized)
        
    cap.release()
    out.release() 