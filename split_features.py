import numpy as np
import os
import argparse
from tqdm import tqdm

def split_and_save(input_file, output_dir_pose, output_dir_hands, output_dir_mouth):
    """
    Loads a .npy file, splits it into pose, hands, and mouth features,
    and saves them to their respective directories.
    """
    try:
        # Load the concatenated data
        data = np.load(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    # --- Feature Dimensions ---
    # These are based on the information you provided.
    # If the concatenation order is different, you only need to change this part.
    pose_dim = 99   # 33 keypoints * 3 (x,y,z)
    hands_dim = 126 # 21 keypoints * 2 hands * 3 (x,y,z)
    
    # --- Slicing based on assumed order: [Pose, Hands, Mouth] ---
    try:
        pose_data = data[:, :pose_dim]
        hands_data = data[:, pose_dim : pose_dim + hands_dim]
        mouth_data = data[:, pose_dim + hands_dim :]
    except IndexError as e:
        print(f"Error slicing {input_file} with shape {data.shape}: {e}")
        print("Please check if the feature dimensions and their order are correct.")
        return

    # Get the original filename to use for the split files
    filename = os.path.basename(input_file)

    # Save the split feature arrays
    np.save(os.path.join(output_dir_pose, filename), pose_data)
    np.save(os.path.join(output_dir_hands, filename), hands_data)
    np.save(os.path.join(output_dir_mouth, filename), mouth_data)

def process_directory(input_dir, output_dir):
    """
    Processes a whole directory (e.g., 'train', 'val', or 'test'),
    creating subdirectories for split features and processing all .npy files.
    """
    # Define and create the output subdirectories for each feature type
    output_dir_pose = os.path.join(output_dir, "pose")
    output_dir_hands = os.path.join(output_dir, "hands")
    output_dir_mouth = os.path.join(output_dir, "mouth")

    os.makedirs(output_dir_pose, exist_ok=True)
    os.makedirs(output_dir_hands, exist_ok=True)
    os.makedirs(output_dir_mouth, exist_ok=True)

    # Get a list of all .npy files in the input directory
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not files_to_process:
        print(f"No .npy files found in {input_dir}. Skipping.")
        return

    print(f"Processing {len(files_to_process)} files in {input_dir}...")
    # Use tqdm for a progress bar
    for filename in tqdm(files_to_process, desc=f"Splitting features in {os.path.basename(input_dir)}"):
        input_file_path = os.path.join(input_dir, filename)
        split_and_save(input_file_path, output_dir_pose, output_dir_hands, output_dir_mouth)
    print(f"Finished processing {input_dir}.")

def main():
    """
    Main function to parse arguments and process the data splits.
    """
    parser = argparse.ArgumentParser(description="Split concatenated mediapipe features into pose, hands, and mouth components.")
    parser.add_argument("--input_root", required=True, type=str,
                        help="Root directory containing the 'train', 'val', 'test' folders with concatenated .npy files.")
    parser.add_argument("--output_root", required=True, type=str,
                        help="Root directory where the split features will be saved into 'pose', 'hands', and 'mouth' subfolders.")
    
    args = parser.parse_args()

    # Process each data split ('train', 'val', 'test')
    for split in ["train", "val", "test"]:
        input_split_dir = os.path.join(args.input_root, split)
        output_split_dir = os.path.join(args.output_root, split)

        if os.path.exists(input_split_dir):
            print(f"--- Processing '{split}' set ---")
            os.makedirs(output_split_dir, exist_ok=True)
            process_directory(input_split_dir, output_split_dir)
        else:
            print(f"Warning: Directory not found for split '{split}': {input_split_dir}")

if __name__ == "__main__":
    main() 