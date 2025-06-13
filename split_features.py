import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd

def split_and_save(input_file, output_dir_pose, output_dir_hands, output_dir_face):
    """
    Loads a .npy file, splits it into pose, hands, and face features based on OpenPose format,
    and saves them to their respective directories.
    """
    try:
        # Load the concatenated data
        data = np.load(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # --- OpenPose Feature Dimensions (Example) ---
    # TODO: Please verify these dimensions based on your actual OpenPose feature extraction.
    # pose_dim = 99   # 33 keypoints * 3 (x,y,z)
    pose_dim = 75   # Example: 25 keypoints * 3 (x, y, conf)
    hands_dim = 126 # Example: 21 keypoints * 2 hands * 3 (x, y, conf)
    
    # The rest is assumed to be face data.
    # --- Slicing based on assumed order: [Pose, Hands, Face] ---
    try:
        pose_data = data[:, :pose_dim]
        hands_data = data[:, pose_dim : pose_dim + hands_dim]
        face_data = data[:, pose_dim + hands_dim :]
    except IndexError as e:
        print(f"Error slicing {input_file} with shape {data.shape}: {e}")
        print("Please check if the feature dimensions and their order are correct.")
        return

    # Get the original filename to use for the split files
    filename = os.path.basename(input_file)

    # Save the split feature arrays
    np.save(os.path.join(output_dir_pose, filename), pose_data)
    np.save(os.path.join(output_dir_hands, filename), hands_data)
    np.save(os.path.join(output_dir_face, filename), face_data)

def process_files_from_csv(csv_path, features_root, output_root):
    """
    Processes features based on entries in a CSV file.
    The CSV tells us which feature files (e.g., from SENTENCE_NAME) to process.
    """
    try:
        df = pd.read_csv(csv_path, sep='\\t', engine='python')
    except FileNotFoundError:
        print(f"Annotation file not found: {csv_path}. Skipping.")
        return

    # Define and create the output subdirectories for each feature type
    output_dir_pose = os.path.join(output_root, "pose")
    output_dir_hands = os.path.join(output_root, "hands")
    output_dir_face = os.path.join(output_root, "face")

    os.makedirs(output_dir_pose, exist_ok=True)
    os.makedirs(output_dir_hands, exist_ok=True)
    os.makedirs(output_dir_face, exist_ok=True)

    print(f"Processing features listed in {csv_path}...")
    
    for sentence_name in tqdm(df['SENTENCE_NAME'].unique(), desc=f"Splitting features for {os.path.basename(csv_path)}"):
        input_file_path = os.path.join(features_root, f"{sentence_name}.npy")
        if os.path.exists(input_file_path):
            split_and_save(input_file_path, output_dir_pose, output_dir_hands, output_dir_face)
        else:
            print(f"Warning: Feature file not found: {input_file_path}")

def main():
    """
    Main function to parse arguments and process the data splits based on CSV files.
    """
    parser = argparse.ArgumentParser(description="Split concatenated OpenPose features based on CSV annotations.")
    parser.add_argument("--csv_root", required=True, type=str,
                        help="Root directory of the CSV annotation files (e.g., '~/asic-3/input_data/csv_data').")
    parser.add_argument("--features_root", required=True, type=str,
                        help="Root directory of the raw, concatenated .npy feature files.")
    parser.add_argument("--output_root", required=True, type=str,
                        help="Root directory where the split features will be saved into 'pose', 'hands', and 'face' subfolders.")
    
    args = parser.parse_args()

    # Process each data split ('train', 'val', 'test')
    # Expand user paths
    csv_root = os.path.expanduser(args.csv_root)
    features_root = os.path.expanduser(args.features_root)
    output_root = os.path.expanduser(args.output_root)

    # Process each data split based on the corresponding CSV file
    for split in ["train", "val", "test"]:
        csv_file = os.path.join(csv_root, f"how2sign_realigned_{split}.csv")
        split_output_dir = os.path.join(output_root, split)

        if os.path.exists(csv_file):
            print(f"--- Processing '{split}' set ---")
            os.makedirs(split_output_dir, exist_ok=True)
            process_files_from_csv(csv_file, features_root, split_output_dir)
        else:
            print(f"Warning: CSV file not found for split '{split}': {csv_file}")

if __name__ == "__main__":
    main() 