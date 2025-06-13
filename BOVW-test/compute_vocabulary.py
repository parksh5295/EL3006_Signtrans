from sklearn.cluster import MiniBatchKMeans
import numpy as np 
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='python3 compute_vocabulary.py --video_dir [VIDEO_DIR] --out_dir [OUT_DIR] --save_model_path [SAVE_MODEL]')
parser.add_argument('--array_dir', type=str, help='directory of videos', required=True)
parser.add_argument('--out_dir', type=str, help='directory to store the outputs', required=True)
parser.add_argument('--n_clusters', type=int, default=10000, help='number of vocab')
args = parser.parse_args()

list_array_paths = os.listdir(args.array_dir)
n_videos = len(list_array_paths)
X = []
# Array containing the number of frames of each video
n_frames_list = []
# Array containing the n_region_proposals of each frame of each video
n_region_proposals_list = []

# Iterate over each video's dope features
for array_path in list_array_paths:
    # Open dope features of each frames of one video
    src_path = os.path.join(args.array_dir, array_path)
    # dope_features_list should be a list of lenth n_frames
    # an element of dope_feature_list is an array of dim (n_region_proposals, 2048)
    dope_features_list = np.load(src_path, allow_pickle=True)
    n_frames = len(dope_features_list)

    X.append(np.vstack(dope_features_list))
    n_frames_list.append(n_frames)
    n_region_proposals = [dope_features.shape[0] for dope_features in dope_features_list]
    n_region_proposals_list.append(n_region_proposals)

# Concatenate the dope features for each frame of each videos
X = np.vstack(X) # (number of total frames, 512)
kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=500,
                random_state=42, verbose=1)
kmeans.fit(X)
np.save(os.path.join(args.out_dir, 'vocab'), kmeans.cluster_centers_)

from sklearn.neighbors import KDTree
tree = KDTree(np.load(os.path.join(args.out_dir, 'vocab.npy')))
start = 0
hist_per_video = []
from tqdm import tqdm
for i in tqdm(range(n_videos)):
    hist_per_frame = []

    n_frames = n_frames_list[i]
    for frame_idx in range(n_frames):
        n_region_proposals = n_region_proposals_list[i][frame_idx]
        end=start+n_region_proposals
        _, clusters = tree.query(X[start:end], k=1)
        hist, _ = np.histogram(clusters.ravel(), bins=args.n_clusters)
        hist_per_frame.append(hist)
        start=end
    hist_per_video.append(hist_per_frame)
hist_for_each_frame = np.concatenate(hist_per_video, axis=0)
documentsPerFeature = np.zeros(args.n_clusters)
for i in tqdm(range(args.n_clusters)):
    documentsPerFeature[i] = (hist_for_each_frame[:, i] != 0).sum()
idf = np.log(hist_for_each_frame.shape[0] / documentsPerFeature)
np.save(os.path.join(args.out_dir, 'idf'), idf)
