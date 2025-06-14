import pandas as pd
from datasets import load_dataset, DatasetDict
import os

def explore_how2sign_data():
    """
    This function explores the Hugging Face How2Sign keypoints dataset and
    the local CSV annotations to understand their structure and find a
    way to map them.
    """
    print("--- Starting Data Exploration ---")

    # --- 1. Explore the Hugging Face Dataset ---
    try:
        print("\n[1/3] Loading Hugging Face dataset 'Saintbook/how2sign_keypoints'...")
        # Load a small portion to avoid long downloads
        hf_dataset = load_dataset("Saintbook/how2sign_keypoints", split="validation", streaming=True)
        
        print("Successfully loaded the dataset stream.")
        
        # Get the first example to inspect its structure
        first_example = next(iter(hf_dataset))
        
        print("\n--- Hugging Face Dataset Info ---")
        print(f"Features (columns): {list(first_example.keys())}")
        print("\nFirst example from Hugging Face dataset:")
        for key, value in first_example.items():
            if key == 'keypoints':
                # Avoid printing the large keypoints array
                print(f"  - {key}: (Numpy array of shape {value.shape})")
            else:
                print(f"  - {key}: {value}")

    except Exception as e:
        print(f"\nERROR: Could not load or inspect the Hugging Face dataset. Reason: {e}")
        return

    # --- 2. Explore the Local CSV Annotation File ---
    try:
        # Define the path to a local CSV file
        # Using validation set as it's smaller and matches the HF dataset split above
        csv_root = "/home/work/asic-3/input_data/csv_data"
        csv_path = os.path.join(csv_root, "how2sign_realigned_val.csv")
        
        print(f"\n[2/3] Reading local CSV file: {csv_path}...")
        
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {csv_path}")
            # Try the train file as a fallback
            csv_path = os.path.join(csv_root, "how2sign_realigned_train.csv")
            print(f"Attempting to read train CSV instead: {csv_path}")
            if not os.path.exists(csv_path):
                 print(f"ERROR: Train CSV also not found. Aborting CSV exploration.")
                 return

        df = pd.read_csv(csv_path, sep='\\t', engine='python')
        
        print("\n--- Local CSV Annotation Info ---")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 rows of the CSV file:")
        print(df.head().to_string())

    except Exception as e:
        print(f"\nERROR: Could not read or inspect the local CSV file. Reason: {e}")
        return
        
    print("\n[3/3] Exploration finished.")
    print("\n--- Next Steps ---")
    print("Please review the outputs above. The key is to find a common column or a relationship")
    print("between the 'Hugging Face Dataset Info' and the 'Local CSV Annotation Info'.")
    print("For example, does the '__key__' from Hugging Face relate to any column in the CSV?")

if __name__ == "__main__":
    explore_how2sign_data() 