import datasets
import tarfile
import numpy as np
import pandas as pd
import os
import logging
import io

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
        downloaded_archives = dl_manager.download(keypoint_urls)

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
                    "keypoints_archive": downloaded_archives["train"],
                    "csv_file": local_csv_paths["train"],
                    "dl_manager": dl_manager,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "keypoints_archive": downloaded_archives["validation"],
                    "csv_file": local_csv_paths["validation"],
                    "dl_manager": dl_manager,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "keypoints_archive": downloaded_archives["test"],
                    "csv_file": local_csv_paths["test"],
                    "dl_manager": dl_manager,
                },
            ),
        ]

    def _generate_examples(self, keypoints_archive, csv_file, dl_manager):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Reading CSV file from {csv_file}")
        df = pd.read_csv(csv_file, sep='\\t', engine='python')
        csv_sentence_names = set(df['SENTENCE_NAME'])
        logging.info(f"First 5 SENTENCE_NAME entries from CSV:\n{df['SENTENCE_NAME'].head().to_string()}")

        npy_filenames_found = []
        matched_count = 0

        for path, file in dl_manager.iter_archive(keypoints_archive):
            if path.endswith(".tar.gz"):
                nested_archive_content = io.BytesIO(file.read())
                with tarfile.open(fileobj=nested_archive_content) as nested_tar:
                    for member in nested_tar.getmembers():
                        if member.name.endswith('.npy'):
                            if len(npy_filenames_found) < 5:
                                npy_filenames_found.append(os.path.basename(member.name))

                            sentence_name = os.path.basename(member.name).replace('.npy', '')
                            
                            if sentence_name in csv_sentence_names:
                                matched_count += 1
                                row = df[df['SENTENCE_NAME'] == sentence_name].iloc[0]
                                text = row['SENTENCE']
                                
                                npy_file = nested_tar.extractfile(member)
                                keypoints_array = np.load(io.BytesIO(npy_file.read()))
                                
                                yield sentence_name, {
                                    "sentence_name": sentence_name,
                                    "keypoints": keypoints_array.tolist(),
                                    "text": text,
                                }

        if matched_count == 0:
            logging.warning("--- DEBUGGING: NO MATCHES FOUND ---")
            logging.warning(f"Could not find any matches between .npy files and the CSV.")
            logging.warning(f"Example .npy filenames found in archive: {npy_filenames_found}")
            logging.warning("Please compare the format of the .npy filenames above with the CSV names printed at the start.") 