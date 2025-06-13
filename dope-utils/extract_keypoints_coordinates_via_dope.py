import sys, os
import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor

from dope.model import dope_resnet50, num_joints
import dope.postprocess as postprocess

import argparse
from tqdm import tqdm

def load_model(modelname='DOPE_v1_0_0', postprocessing = 'ppi'):
    """
        Load DOPE model given modelname.
    """

    # load model
    ckpt_fname = osp.join(_thisdir, 'models', modelname+'.pth.tgz')
    if not os.path.isfile(ckpt_fname):
        raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
    print('Loading model', modelname)
    ckpt = torch.load(ckpt_fname, map_location=device)
    #ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])
    if ckpt['half']: model = model.half()
    model = model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    return model, ckpt

def dope(model, pil_image, ckpt, postprocessing = 'ppi'):
    """
        Apply the DOPE model to pil_image.
        Process the results into dictionnary.
    """
    imlist = [ToTensor()(pil_image).to(device)]
    if ckpt['half']: imlist = [im.half() for im in imlist]
    resolution = imlist[0].size()[-2:]

    # forward pass of the dope network
   #  print('Running DOPE')
    with torch.no_grad():
        results = model(imlist, None)[0]

    # postprocess results (pose proposals integration, wrists/head assignment)
    # print('Postprocessing')
    assert postprocessing in ['nms','ppi']
    parts = ['body','hand','face']
    if postprocessing=='ppi':
        res = {k: v.float().data.cpu().numpy() for k,v in results.items()}
        detections = {}
        for part in parts:
            detections[part] = LCRNet_PPI_improved(res[part+'_scores'], res['boxes'], res[part+'_pose2d'], res[part+'_pose3d'], resolution, **ckpt[part+'_ppi_kwargs'])
    else: # nms
        detections = {}
        for part in parts:
            dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
            dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
            detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
            if part=='hand':
                for i in range(len(detections[part])): 
                    detections[part][i]['hand_isright'] = bestcls<ckpt['hand_ppi_kwargs']['K']

    # assignment of hands and head to body
    detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)

   # Output
    det_poses2d = {part+'_pose2d': np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections)>0 else np.empty( (0,num_joints[part],2), dtype=np.float32) for part, part_detections in detections.items()}
    det_poses3d = {part+'_pose3d': np.stack([d['pose3d'] for d in part_detections], axis=0) if len(part_detections)>0 else np.empty( (0,num_joints[part],3), dtype=np.float32) for part, part_detections in detections.items()}

    return {**det_poses2d, **det_poses3d}

def process_video(filename, ckpt):
    """
        Process a video given its filename.
    """

    # Process video
    cap = cv2.VideoCapture(filename)
    # Pointer to current_frame
    current_frame = 0
    # Video length
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Key points coordinates
    results = {}
    results['body_pose2d'] = np.zeros((video_length, 1, 13, 2))
    results['body_pose3d'] = np.zeros((video_length, 1, 13, 3))

    results['hand_pose2d'] = np.zeros((video_length, 2, 21, 2))
    results['hand_pose3d'] = np.zeros((video_length, 2, 21, 3))

    results['face_pose2d'] = np.zeros((video_length, 1, 84, 2))
    results['face_pose3d'] = np.zeros((video_length, 1, 84, 3))

    parts = ['body', 
            'hand',
            'face']
    poses = ['pose2d',
            'pose3d']

    # We keep indexes of faillure case
    missing_value = {}
    for part in parts:
        for pose in poses:
            missing_value[part + '_' + pose] = 0

    # Iterate over the frames
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert to PIL so we don't modify existing dope codes.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        current_results = dope(model, frame_pil, ckpt)

        # Iterate over body parts, to format the outputs
        for part in parts:
            for pose in poses:
                key = part + '_' + pose

                # If body part is detected
                if current_results[key].size != 0:
                    # Ugly hack, in order to match the output size
                    if current_results[key].shape[0] > results[key][current_frame].shape[0]:
                        results[key][current_frame] = current_results[key][[results[key][current_frame].shape[0]]]
                    else:
                        results[key][current_frame] = current_results[key]
                    # We fill missing value with the nearest valid outputs
                    if missing_value[key] > 0:
                        for fill_missing_value_idx in range(missing_value[key]):
                            results[key][fill_missing_value_idx] = results[key][current_frame]
                    missing_value[key] = -1
                # O.W, we interpolate with last values
                else:
                    if missing_value[key] != - 1:
                        missing_value[key] += 1
                    else:
                        results[key][current_frame] = results[key][current_frame-1]
        current_frame += 1
        if current_frame == video_length:
            break
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='python3 extract_keypoints_coordinates_via_dope.py --video_dir [VIDEO_DIR] --out_dir [OUT_DIR]')
    parser.add_argument('--video_dir', type=str, help='directory of videos', required=True)
    parser.add_argument('--out_dir', type=str, help='directory to store the outputs', required=True)
    args = parser.parse_args()

    # Useful ugly global variables
    _thisdir = 'dope'
    postprocessing = 'ppi'

    if postprocessing=='ppi':
        sys.path.append( _thisdir+'/lcrnet-v2-improved-ppi/')
    try:
        from lcr_net_ppi_improved import LCRNet_PPI_improved
    except ModuleNotFoundError:
        raise Exception('To use the pose proposals integration (ppi) as postprocessing, please follow the readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, you can use --postprocess nms without any installation, with a slight decrease of performance.')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parts = ['body', 
            'hand',
            'face']
    poses = ['pose2d',
            'pose3d']

    # Create directories to save outputs
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for part in parts:
        for pose in poses:
            key = part + '_' + pose
            sub_out_dir = os.path.join(args.out_dir, key)
            if not os.path.exists(sub_out_dir):
                os.makedirs(sub_out_dir)

    # Create model
    model, ckpt = load_model(modelname='DOPE_v1_0_0', postprocessing = 'ppi')

    # Iterate over array of videos
    for video_name in tqdm(os.listdir(args.video_dir)):
        # Process one video at a time
        filename = os.path.join(args.video_dir, video_name)
        out = process_video(filename, ckpt)

        # Save features
        for part in parts:
            for pose in poses:
                key = part + '_' + pose
                sub_out_dir = os.path.join(args.out_dir, key)
                np.save(os.path.join(sub_out_dir, video_name[:-4]), out[key])
