import os
import cv2
from tqdm import tqdm
from glob import glob

from inference.process import Processor
from head import HeadFaceModel
from hpe import HeadPoseModel


SOURCE = 'videos'
OUTPUT_DIR = 'outputs/videos' 
BZ = 1



if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    headface = HeadFaceModel('weights/yolov7-headface-v1.onnx')
    headpose = HeadPoseModel('weights/cmu.onnx')
    
    processor = Processor(
        headmodel=headface, headposemodel=headpose
    )
    
    for video_path in tqdm(glob(f"{SOURCE}/*")):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        w, h, fps = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)),
        )
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(
            f"{OUTPUT_DIR}/{video_path.split('/')[-1].split('.')[0]}_predict.avi",
            fourcc,
            fps,
            (w, h),
        )
        frames = []
        for _ in tqdm(range(total_frames), desc='Procssing time'):
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            if len(frames) == BZ:
                frames = processor(frames)
                for frame in frames:
                    out.write(frame)
                frames = []

        if len(frames) > 0:
            frames = processor(frames)
            for frame in frames:
                out.write(frame)
        
        cap.release()
        out.release()