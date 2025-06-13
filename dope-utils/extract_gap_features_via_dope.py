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

# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py

def forward_features(self, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                            " Found invalid box {} for target at index {}."
                            .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        # Add global average pooling layer
        gap = torch.nn.AdaptiveAvgPool2d((1, 1)) 
        return gap(features[0]).view(-1).detach().cpu().numpy()

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
    results = np.zeros((video_length, 1024))

    # Iterate over the frames
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert to PIL so we don't modify existing dope codes.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        imlist = [ToTensor()(frame_pil).to(device)]
        if ckpt['half']: imlist = [im.half() for im in imlist]
        resolution = imlist[0].size()[-2:]

        # forward pass of the dope network
        with torch.no_grad():
            results[current_frame, :] = forward_features(model, imlist)

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

    # Create directories to save outputs
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Create model
    model, ckpt = load_model(modelname='DOPE_v1_0_0', postprocessing = 'ppi')

    model.eval()

    # Iterate over array of videos
    for video_name in tqdm(os.listdir(args.video_dir)):
        # Process one video at a time
        filename = os.path.join(args.video_dir, video_name)
        out = process_video(filename, ckpt)

        np.save(os.path.join(args.out_dir, video_name[:-4]), out)
