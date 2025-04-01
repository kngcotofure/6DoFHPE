import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))

import cv2
import numpy as np
import utils
from head import HeadFaceModel
from hpe import HeadPoseModel


class Processor:
    def __init__(
        self,
        headmodel,
        headposemodel
    ):
        self.headmodel = headmodel
        self.headposemodel = headposemodel
        
    def __call__(
        self,
        frames
    ):
        # head detection
        heads = self.headmodel.batch_predict(frames, det_thres=0.8)
        
        _bboxes, counts = [], []
        for row in heads:
            counts.append(len(row[0]))
            _bboxes.extend(row[0])
            
        _frames = []
        for idx in range(len(frames)):
            _frames.extend([frames[idx]] * counts[idx])
        
        if len(_bboxes) != 0:
            # head pose estimation
            results, configs = self.headposemodel.batch_predict(
                imgs = _frames,
                bboxes= _bboxes,
                return_dict=True
            )
            
            index = 0
            for idx in range(len(frames)):
                group_size = counts[idx]
                for _ in range(group_size):
                    if index < len(results):
                        self.visualize(frames[idx], results[index], configs[index])
                        index += 1
        
        return frames
    
    def visualize(self, frame, result, config):
       
        p_deg, y_deg, r_deg = result

        utils.draw_axis(
            frame, 
            y_deg, 
            p_deg, 
            r_deg, 
            config['x_min'] + int(.5*(config['x_max']-config['x_min'])), 
            config['y_min'] + int(.5*(config['y_max']-config['y_min'])), 
            size=config['bbox_width']
        )
        
        pitch = p_deg * np.pi / 180
        yaw = -(y_deg * np.pi / 180)
        roll = r_deg * np.pi / 180
        
        label = "{}:{:.2f}, {}:{:.2f}, {}:{:.2f} ".format('x', pitch, 'y',yaw,'z', roll)
        y = config['y_min']
        cv2.putText(frame, label, (config['x_min'], y), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
        cv2.rectangle(frame, (config['x_min'], config['y_min']), (config['x_max'], config['y_max']), [0, 0, 255], 2)



if __name__ == '__main__':
    import glob
    
    headface = HeadFaceModel('weights/yolov7-headface-v1.onnx')
    headpose = HeadPoseModel('weights/cmu.onnx')
    
    imgs = [cv2.imread(img) for img in glob.glob("images/*")]
    
    processor = Processor(
        headmodel=headface, headposemodel=headpose
    )
    
    results = processor(imgs)
    os.makedirs("outputs/images", exist_ok=True)
    for idx, frame in enumerate(results):
        cv2.imwrite(f"outputs/images/image_{idx}.jpg", frame)
    