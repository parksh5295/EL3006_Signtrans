import datasets
import tarfile
import numpy as np
import pandas as pd
import os
import logging

_DESCRIPTION = "How2Sign 2D Keypoints dataset"
_HOMEPAGE = "https://github.com/how2sign/how2sign"
_LICENSE = "cc-by-nc-4.0"
_CITATION = """@inproceedings{duarte2021how2sign,
 title={How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language},
 author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Zisserman, Andrew and Metze, Florian and Cook, C. and De la Torre, F.},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2021}
}"""

class How2SignKeypoints(datasets.GeneratorBasedBuilder):
    """How2Sign 2D Keypoints Dataset Loader"""

    # URL for the remote repository containing the keypoint archives
    _BASE_URL = "https://huggingface.co/datasets/Saintbook/how2sign_keypoints/resolve/main"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "sentence_name": datasets.Value("string"),
                "keypoints": datasets.features.Sequence(
                    datasets.features.Sequence(datasets.Value("float32"))
                ),
                "text": datasets.Value("string"),
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # 1. Download and extract keypoint archives from the remote URL
        keypoint_urls = {
            "train": f"{self._BASE_URL}/train_2D_keypoints.tar.gz",
            "validation": f"{self._BASE_URL}/val_2D_keypoints.tar.gz",
            "test": f"{self._BASE_URL}/test_2D_keypoints.tar.gz",
        }
        keypoints_dirs = dl_manager.download_and_extract(keypoint_urls)

        # 2. Define the paths to the LOCAL CSV files
        # These paths should match the environment where the code is run
        csv_root = "/home/work/asic-3/input_data/csv_data"
        local_csv_paths = {
            "train": os.path.join(csv_root, "how2sign_realigned_train.csv"),
            "validation": os.path.join(csv_root, "how2sign_realigned_val.csv"),
            "test": os.path.join(csv_root, "how2sign_realigned_test.csv"),
        }
        
        # Verify that local CSV files exist
        for split, path in local_csv_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Local CSV file for '{split}' split not found at {path}. "
                    "Please ensure the CSV files are in the correct directory."
                )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "keypoints_dir": keypoints_dirs["train"],
                    "csv_file": local_csv_paths["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "keypoints_dir": keypoints_dirs["validation"],
                    "csv_file": local_csv_paths["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "keypoints_dir": keypoints_dirs["test"],
                    "csv_file": local_csv_paths["test"],
                },
            ),
        ]

    def _generate_examples(self, keypoints_dir, csv_file):
        logging.basicConfig(level=logging.INFO)
        logging.info(f"--- Starting data generation for split ---")
        logging.info(f"Attempting to read keypoints from: {keypoints_dir}")
        logging.info(f"Attempting to read annotations from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file, sep='\\t', engine='python')
            logging.info(f"Successfully loaded CSV. Columns: {df.columns.tolist()}")
            logging.info(f"First 5 SENTENCE_NAME entries from CSV:\n{df['SENTENCE_NAME'].head().to_string()}")
        except Exception as e:
            logging.error(f"Failed to load or process CSV file: {e}")
            return

        found_npy_files_count = 0
        matched_files_count = 0
        npy_file_examples = []
        
        for root, _, files in os.walk(keypoints_dir):
            for f_name in files:
                if f_name.endswith('.npy'):
                    found_npy_files_count += 1
                    if len(npy_file_examples) < 5:
                        npy_file_examples.append(f_name)

                    sentence_name = f_name.replace('.npy', '')
                    
                    row = df[df['SENTENCE_NAME'] == sentence_name]
                    if not row.empty:
                        matched_files_count += 1
                        text = row.iloc[0]['SENTENCE']
                        
                        keypoint_file_path = os.path.join(root, f_name)
                        keypoints_array = np.load(keypoint_file_path)
                        
                        yield sentence_name, {
                            "sentence_name": sentence_name,
                            "keypoints": keypoints_array.tolist(),
                            "text": text,
                        }
        
        logging.warning(f"--- Generation summary ---")
        logging.warning(f"Total .npy files found: {found_npy_files_count}")
        logging.warning(f"Total files matched with CSV: {matched_files_count}")
        if found_npy_files_count > 0:
            logging.warning(f"First 5 .npy filenames found: {npy_file_examples}") 