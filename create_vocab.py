import os
import pandas as pd
from collections import Counter
from typing import List, Set
from datasets import load_dataset, DatasetDict, concatenate_datasets

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

def main_huggingface(
    hf_dataset_id: str,
    text_col: str,
    gloss_col: str,
    output_dir: str,
    num_proc: int
):
    """
    Generate vocabularies from a Hugging Face dataset in parallel.
    """
    print(f"Loading dataset '{hf_dataset_id}' from Hugging Face Hub...")
    all_splits: DatasetDict = load_dataset(hf_dataset_id)

    # Combine all splits (train, validation, test) to build a comprehensive vocab
    combined_dataset = concatenate_datasets([all_splits[s] for s in all_splits.keys()])

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
    print(f"Processing glosses and texts in parallel with {num_proc} processes...")
    
    def tokenize_sent(batch):
        return {"tokens": [s.split() for s in batch if s]}

    # Process glosses
    gloss_tokens_dataset = combined_dataset.map(
        lambda batch: tokenize_sent(batch[gloss_col]),
        batched=True,
        num_proc=num_proc,
        remove_columns=combined_dataset.column_names
    )
    all_glosses = [token for example in gloss_tokens_dataset for token in example['tokens']]
    
    # Process texts
    text_tokens_dataset = combined_dataset.map(
        lambda batch: tokenize_sent(batch[text_col]),
        batched=True,
        num_proc=num_proc,
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

    main_huggingface(
        hf_dataset_id="Saintbook/how2sign_keypoints",
        gloss_col="SENTENCE",
        text_col="SENTENCE",
        output_dir="data/openpose",
        num_proc=num_cpus
    )
