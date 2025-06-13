import os
import pandas as pd
from collections import Counter
from typing import List, Set

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

def process_files(
    data_dir: str,
    train_file: str,
    val_file: str,
    test_file: str,
    gloss_col: str,
    text_col: str,
    separator: str,
    output_dir: str,
):
    """Generic function to process annotation files and create vocabularies."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define column types
    dtype_dict = {gloss_col: str, text_col: str}

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

    # Combine all data
    all_glosses = []
    all_texts = []

    for df in [train_df, val_df, test_df]:
        # Convert to string and handle NaN values
        glosses = df[gloss_col].fillna("").astype(str)
        translations = df[text_col].fillna("").astype(str)

        all_glosses.extend(glosses.str.split().tolist())
        all_texts.extend(translations.str.split().tolist())

    # Flatten lists
    all_glosses = [token for sent in all_glosses for token in sent]
    all_texts = [token for sent in all_texts for token in sent]

    # Create vocabulary files
    create_vocab_file(all_glosses, os.path.join(output_dir, "gloss.vocab"))
    create_vocab_file(all_texts, os.path.join(output_dir, "text.vocab"))


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


if __name__ == "__main__":
    # Select which main function to run.
    # To run for OpenPose CSV files, change the function call below.
    # main_how2sign_tsv()
    main_openpose_csv() 