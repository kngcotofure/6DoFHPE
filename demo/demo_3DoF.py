import os
import cv2
import sys
import time
import argparse
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
from torch.backends import cudnn
from torchvision import transforms

import utils
from model import RepNet6D
from hdssd import Head_detection

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))
from inference.head import HeadFaceModel

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--output', type=str, default='output',
                        help='Saved result path')
    parser.add_argument('--source', help='Image source or Video source',
                        default='./video.mp4')
    parser.add_argument('--snapshot-model', help='Name of model snapshot.',
                        default='./cmu.pth', type=str)
    parser.add_argument('--head-model',
                        help='Name of head detection model', 
                        default='', type=str)
    parser.add_argument('--device', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    cudnn.enabled = True
    gpu = args.device
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    print("\n\ndevice: ", device, "\n\n")
    transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    Head_detection = Head_detection(args.head_model)
    
    # cam = args.cam_id
    snapshot_path = args.snapshot_model
    model = RepNet6D(backbone_name='RepVGG-B1g4',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
    # Load snapshot
    saved_state_dict = torch.load(os.path.join(snapshot_path), map_location=None if torch.cuda.is_available() else 'cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    
    model.to(device)
    model.eval()

    for video_path in tqdm(glob(f"{args.source}/*")):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        w, h, fps = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)),
        )
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(
            f"{args.output}/{video_path.split('/')[-1].split('.')[0]}_predict.avi",
            fourcc,
            fps,
            (w, h),
        )
        
        for _ in tqdm(range(total_frames), desc='Procssing time'):
            ret, frame = cap.read()
            if not ret:
                break
            
            (h, w, c) = frame.shape
            t0 = time.time()
            frame, heads = Head_detection.run(frame, w, h)
            t1 = time.time()

            for head in heads:
                x_min, y_min = int(head['left']), int(head['top'])
                x_max, y_max = int(head['right']), int(head['bottom'])
                bbox_width, bbox_height = abs(x_max - x_min), abs(y_max - y_min)
                
                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)
                
                # processing image crop
                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)
                img = torch.Tensor(img[None, :]).to(device)
                
                # start = time.time()
                t2 = time.time()
                R_pred = model(img) # model HPE predict
                t3 = time.time()
  
                # end = time.time()
                # print('Head pose estimation: %2f ms' % ((end - start)*1000.))
                
                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                
                p_pred_deg = euler[:, 0].detach().cpu()
                y_pred_deg = euler[:, 1].detach().cpu()
                r_pred_deg = euler[:, 2].detach().cpu()
                
                utils.draw_axis(frame, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
                
                pitch = p_pred_deg[0] * np.pi / 180
                yaw = -(y_pred_deg[0] * np.pi / 180)
                roll = r_pred_deg[0] * np.pi / 180
                label = "{}: {:.2f}, {}: {:.2f}, {}: {:.2f} ".format('x', pitch, 'y',yaw,'z', roll)
                y = y_min - 5 if y_min - 5 > 5 else y_min + 5
                cv2.putText(frame, label, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
            t4 = time.time()
            out.write(frame)
            
        cap.release()
        out.release()
