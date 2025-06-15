import pandas as pd
from datasets import load_dataset
import os
import sys

def explore_how2sign_data():
    """
    This function explores the How2Sign dataset and saves the output to a file
    to avoid terminal buffer issues.
    """
    output_filename = "exploration_log.txt"
    
    # Redirect stdout to the output file
    original_stdout = sys.stdout
    with open(output_filename, 'w', encoding='utf-8') as f:
        sys.stdout = f

        print("--- Starting Data Exploration ---")

        # --- 1. Explore the Hugging Face Dataset ---
        try:
            print("\n[1/3] Loading Hugging Face dataset 'Saintbook/how2sign_keypoints'...")
            # Use streaming to avoid downloading the whole dataset
            hf_dataset = load_dataset("Saintbook/how2sign_keypoints", split="train", streaming=True)
            print("Successfully loaded the dataset stream.")
            
            # Get the first example to inspect its structure
            first_example = next(iter(hf_dataset))
            
            print("\n--- Hugging Face Dataset Info ---")
            print(f"Features (columns): {list(first_example.keys())}")
            
            print("\nFirst example from Hugging Face dataset:")
            for key, value in first_example.items():
                # Avoid printing large binary/array data directly
                if hasattr(value, 'shape'):
                    print(f"  - {key}: (Numpy array of shape {value.shape})")
                elif isinstance(value, bytes):
                    print(f"  - {key}: (Binary data of length {len(value)})")
                else:
                    # Use repr() to make special characters visible
                    print(f"  - {key}: {repr(value)}")

        except Exception as e:
            print(f"\nERROR: Could not load or inspect the Hugging Face dataset. Reason: {e}")

        # --- 2. Explore the Local CSV Annotation File (for comparison) ---
        try:
            csv_root = "/home/work/asic-3/input_data/csv_data"
            csv_path = os.path.join(csv_root, "how2sign_realigned_train.csv")
            
            print(f"\n[2/3] Reading local CSV file for context: {csv_path}...")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep='\\t', engine='python')
                print("\n--- Local CSV Annotation Info ---")
                print(f"Columns: {df.columns.tolist()}")
                print("\nFirst 5 rows of the CSV file:")
                print(df.head().to_string())
            else:
                print(f"INFO: CSV file not found at {csv_path}, skipping.")

        except Exception as e:
            print(f"\nERROR: Could not read or inspect the local CSV file. Reason: {e}")
            
        print("\n[3/3] Exploration finished.")

    # Restore original stdout
    sys.stdout = original_stdout
    print(f"Data exploration output has been saved to '{output_filename}'.")


if __name__ == "__main__":
    explore_how2sign_data() 