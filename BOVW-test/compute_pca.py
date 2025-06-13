from sklearn.decomposition import IncrementalPCA
import numpy as np 
import os
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='python3 compute_pca.py --video_dir [VIDEO_DIR] --out_dir [OUT_DIR] --save_model_path [SAVE_MODEL]')
parser.add_argument('--array_dir', type=str, help='directory of videos', required=True)
parser.add_argument('--out_dir', type=str, help='directory to store the outputs', required=True)
parser.add_argument('--save_model_path', type=str, help='path to save model', required=True)
parser.add_argument('--n_components', type=int, default=256, help='dimensionality of PCA reduction')
args = parser.parse_args()

list_array_paths = os.listdir(args.array_dir)
n_videos = len(list_array_paths)
X = []
# Array containing the number of frames of each video
n_frames_list = []
# Array containing the n_region_proposals of each frame of each video
n_region_proposals_list = []

# Iterate over each video's dope features
for array_path in tqdm(list_array_paths):
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
X = np.vstack(X) # (number of total frames, 2048)
print('.')
# Dimensionality reduction 2048 to 512
pca = IncrementalPCA(n_components=args.n_components)
X_reduced = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)

# Mkdir
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

count = 0
# Save features, need to "unflatten" X_reduced
for video_idx, array_path in enumerate(list_array_paths):
    dst_path = os.path.join(args.out_dir, array_path)
    dope_features_reduced_list = [] 
    n_frames = n_frames_list[video_idx]
    # Iterate over each frame
    print(n_frames)
    for frame_idx in range(n_frames):
        n_region_proposals = n_region_proposals_list[video_idx][frame_idx]
        dope_features_reduced = np.zeros((n_region_proposals, args.n_components))
        # For each frame, we have multiple region proposals
        for n_region_proposals_idx in range(n_region_proposals):
            dope_features_reduced[n_region_proposals_idx, :] = X_reduced[count]
            count += 1
        dope_features_reduced_list.append(dope_features_reduced)

    np.save(dst_path, dope_features_reduced_list)

# Save model
with open(args.save_model_path, "wb") as f:
    pickle.dump(pca, f)
