import datasets
import tarfile
import numpy as np
import pandas as pd
import os

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

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description="Default How2Sign keypoints dataset."),
    ]

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
        # Assumes the user has downloaded the data manually and placed it in the cache.
        # This avoids automatic download which might fail for large files.
        # The user should place the files in ~/.cache/huggingface/datasets/downloads/
        keypoints_archive_path = {
            "train": "/home/work/asic-3/input_data/keypoints/train_2D_keypoints.tar.gz",
            "validation": "/home/work/asic-3/input_data/keypoints/val_2D_keypoints.tar.gz",
            "test": "/home/work/asic-3/input_data/keypoints/test_2D_keypoints.tar.gz",
        }
        
        csv_path = {
            "train": "/home/work/asic-3/input_data/csv_data/how2sign_realigned_train.csv",
            "validation": "/home/work/asic-3/input_data/csv_data/how2sign_realigned_val.csv",
            "test": "/home/work/asic-3/input_data/csv_data/how2sign_realigned_test.csv",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "keypoints_archive": keypoints_archive_path["train"],
                    "csv_file": csv_path["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "keypoints_archive": keypoints_archive_path["validation"],
                    "csv_file": csv_path["validation"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "keypoints_archive": keypoints_archive_path["test"],
                    "csv_file": csv_path["test"],
                },
            ),
        ]

    def _generate_examples(self, keypoints_archive, csv_file):
        df = pd.read_csv(csv_file, sep='\\t', engine='python')
        
        with tarfile.open(keypoints_archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.npy'):
                    sentence_name = os.path.basename(member.name).replace('.npy', '')
                    
                    # Find corresponding row in dataframe
                    row = df[df['SENTENCE_NAME'] == sentence_name]
                    if not row.empty:
                        text = row.iloc[0]['SENTENCE']
                        
                        # Extract and load .npy file
                        f = tar.extractfile(member)
                        keypoints_array = np.load(f)
                        
                        yield sentence_name, {
                            "sentence_name": sentence_name,
                            "keypoints": keypoints_array.tolist(),
                            "text": text,
                        } 