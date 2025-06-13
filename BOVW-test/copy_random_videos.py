import os
import numpy as np
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='python3 copy_random_videos.py --video_dir [VIDEO_DIR] --out_dir [OUT_DIR]')
parser.add_argument('--video_dir', type=str, help='directory of videos', required=True)
parser.add_argument('--out_dir', type=str, help='directory to store the outputs', required=True)
parser.add_argument('--n_samples', type=int, default=600, help='number of random videos to move')
parser.add_argument('--seed', type=float, default=42, help='number of random videos to copy')
args = parser.parse_args()

np.random.seed(args.seed)
list_videos = os.listdir(args.video_dir)
n = len(list_videos)

random_idx = np.random.permutation(np.arange(n))[:args.n_samples]
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for idx in tqdm(random_idx):
    src = os.path.join(args.video_dir, list_videos[idx])
    dst = os.path.join(args.out_dir, list_videos[idx])
    shutil.copy(src, dst)
