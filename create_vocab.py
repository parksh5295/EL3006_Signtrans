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

def main():
    # Paths
    data_dir = os.path.expanduser("~/asic-3/input_data/tsv_files_how2sign/tsv_files_how2sign")
    output_dir = "data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read TSV files
    train_df = pd.read_csv(os.path.join(data_dir, "cvpr23.fairseq.i3d.train.how2sign.tsv"), sep="\t")
    val_df = pd.read_csv(os.path.join(data_dir, "cvpr23.fairseq.i3d.val.how2sign.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(data_dir, "cvpr23.fairseq.i3d.test.how2sign.tsv"), sep="\t")
    
    # Combine all data
    all_glosses = []
    all_texts = []
    
    for df in [train_df, val_df, test_df]:
        all_glosses.extend(df["gloss"].str.split().tolist())
        all_texts.extend(df["text"].str.split().tolist())
    
    # Flatten lists
    all_glosses = [token for sent in all_glosses for token in sent]
    all_texts = [token for sent in all_texts for token in sent]
    
    # Create vocabulary files
    create_vocab_file(all_glosses, os.path.join(output_dir, "gloss.vocab"))
    create_vocab_file(all_texts, os.path.join(output_dir, "text.vocab"))

if __name__ == "__main__":
    main() 