import os
import pandas as pd
from collections import Counter
from typing import List, Set
from datasets import Dataset, concatenate_datasets

def create_vocab_file(tokens: List[str], output_file: str, min_freq: int = 1) -> None:
    """Create a vocabulary file from a list of tokens."""
    # Count token frequencies
    counter = Counter(tokens)
    
    # Filter by minimum frequency
    vocab = [token for token, freq in counter.items() if freq >= min_freq]
    
    # Sort by frequency (most frequent first)
    vocab.sort(key=lambda x: counter[x], reverse=True)
    
    # Add special tokens
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab = special_tokens + vocab
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(f"{token}\n")

def _tokenize_batch(batch, column_name: str):
    """
    Helper function to tokenize sentences from a specific column in a batch.
    This is defined at the top level for robust pickling in multiprocessing.
    """
    if column_name not in batch:
        raise KeyError(f"Column '{column_name}' not found in batch. Available: {list(batch.keys())}")
    
    sentences = batch[column_name]
    # Filter out None or empty strings before splitting
    return {"tokens": [s.split() for s in sentences if isinstance(s, str) and s]}

def main_from_csv(
    csv_root: str,
    text_col: str,
    gloss_col: str,
    output_dir: str,
    num_proc: int
):
    """
    Generate vocabularies from local CSV files in parallel.
    print(f"Loading dataset '{hf_dataset_id}' from Hugging Face Hub...")
    all_splits: DatasetDict = load_dataset(hf_dataset_id)
    """
    csv_root = os.path.expanduser(csv_root)
    print(f"Loading CSV files from: {csv_root}")
    
    # Load all splits into pandas dataframes
    splits = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(csv_root, f"how2sign_realigned_{split_name}.csv")
        if os.path.exists(csv_path):
            splits[split_name] = pd.read_csv(csv_path, sep='\\t', engine='python')
        else:
            print(f"Warning: {csv_path} not found. Skipping.")
    
    if not splits:
        raise FileNotFoundError(f"No CSV files found in {csv_root}. Please check the path.")

    # Convert pandas dataframes to Hugging Face Datasets for easy mapping
    # combined_dataset = concatenate_datasets([all_splits[s] for s in all_splits.keys()])
    hf_datasets = {name: Dataset.from_pandas(df) for name, df in splits.items()}
    combined_dataset = concatenate_datasets(list(hf_datasets.values()))
    
    os.makedirs(output_dir, exist_ok=True)
    
    '''
    # Read data files
    train_df = pd.read_csv(
        os.path.join(data_dir, train_file), sep=separator, dtype=dtype_dict
    )
    val_df = pd.read_csv(
        os.path.join(data_dir, val_file), sep=separator, dtype=dtype_dict
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, test_file), sep=separator, dtype=dtype_dict
    )
    '''

    # --- PRE-FLIGHT CHECK: Fail fast if columns are missing ---
    required_cols = {gloss_col, text_col}
    available_cols = set(combined_dataset.column_names)
    if not required_cols.issubset(available_cols):
        missing_cols = required_cols - available_cols
        raise ValueError(
            f"ERROR: The specified columns {list(missing_cols)} were not found in the CSV files.\\n"
            f"Available columns are: {list(available_cols)}"
        )
    # --- End of Check ---

    print(f"Processing glosses and texts in parallel with {num_proc} processes...")
    
    # Process glosses
    gloss_tokens_dataset = combined_dataset.map(
        # lambda batch: tokenize_sent(batch[gloss_col]),
        _tokenize_batch,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"column_name": gloss_col},
        remove_columns=combined_dataset.column_names
    )
    all_glosses = [token for example in gloss_tokens_dataset for token in example['tokens']]
    
    # Process texts
    text_tokens_dataset = combined_dataset.map(
        # lambda batch: tokenize_sent(batch[text_col]),
        _tokenize_batch,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"column_name": text_col},
        remove_columns=combined_dataset.column_names
    )
    all_texts = [token for example in text_tokens_dataset for token in example['tokens']]

    # Create vocabulary files
    print(f"Creating gloss vocabulary at {os.path.join(output_dir, 'gloss.vocab')}...")
    create_vocab_file(all_glosses, os.path.join(output_dir, "gloss.vocab"))
    
    print(f"Creating text vocabulary at {os.path.join(output_dir, 'text.vocab')}...")
    create_vocab_file(all_texts, os.path.join(output_dir, "text.vocab"))
    
    print("Vocabulary creation finished.")

'''
def main_how2sign_tsv():
    """Process original How2Sign TSV files."""
    process_files(
        data_dir=os.path.expanduser(
            "~/asic-3/input_data/tsv_files_how2sign/tsv_files_how2sign"
        ),
        train_file="cvpr23.fairseq.i3d.train.how2sign.tsv",
        val_file="cvpr23.fairseq.i3d.val.how2sign.tsv",
        test_file="cvpr23.fairseq.i3d.test.how2sign.tsv",
        gloss_col="glosses",
        text_col="translation",
        separator="\t",
        output_dir="data/how2sign",
    )


def main_openpose_csv():
    """Process realigned OpenPose CSV files."""
    process_files(
        data_dir=os.path.expanduser("~/asic-3/input_data/csv_data"),
        train_file="how2sign_realigned_train.csv",
        val_file="how2sign_realigned_val.csv",
        test_file="how2sign_realigned_test.csv",
        gloss_col="SENTENCE",  # Assuming gloss is in 'SENTENCE'
        text_col="SENTENCE",  # Assuming text is also in 'SENTENCE'
        separator="\t",
        output_dir="data/openpose",
    )
'''


if __name__ == "__main__":
    # Use a portion of available CPUs for parallel processing
    num_cpus = max(1, os.cpu_count() // 2)

    main_from_csv(
        csv_root="~/asic-3/input_data/csv_data",
        gloss_col="SENTENCE",
        text_col="SENTENCE",
        output_dir="data/openpose",
        num_proc=num_cpus
    )
