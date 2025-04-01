import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",))

import cv2
import numpy as np
import onnxruntime as ort
import utils


class HeadPoseModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
        # Given mean and std values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def load_model(self, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) -> None:
        """
        Example load model
        """
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options, providers)

        self.inps = [inp.name for inp in self.session.get_inputs()]
        self.opts = [opt.name for opt in self.session.get_outputs()]
        _, _, h, w = self.session.get_inputs()[0].shape
        self.model_inpsize = (h, w)

    def predict(self, img: np.array, bbox:list= None, return_dict=False) -> list:
        # preprocess input
        input_img = img.copy()
        w, h, _ = input_img.shape
        configs = dict()
        if bbox is not None:
            configs.update(self.get_bbox(bbox, w, h))
            x_min, y_min = configs['x_min'], configs['y_min']
            x_max, y_max = configs['x_max'], configs['y_max']
            input_img = input_img[y_min:y_max, x_min:x_max]

        # model prediction
        tensor = self.preprocess(input_img)
        if len(tensor.shape) == 4:
            tensor = [tensor]
        outputs = self.session.run(self.opts, dict(zip(self.inps, tensor)))[0]
        
        outputs = [self.postprocess(output) for output in outputs]

        return outputs, configs if return_dict else outputs
    
    def batch_predict(self, imgs: list, bboxes:list= None, return_dict=False) -> list:
        configs, tensors = [], []
        assert len(imgs) == len(bboxes)
   
        for idx, img in enumerate(imgs):
            input_img = img.copy()
            w, h, _ = input_img.shape
            if bboxes is not None:
                config = self.get_bbox(input_img, bboxes[idx], w, h)
                x_min, y_min = config['x_min'], config['y_min']
                x_max, y_max = config['x_max'], config['y_max']
                input_img = input_img[y_min:y_max, x_min:x_max]
                
            tensor = self.preprocess(input_img)
            tensors.append(tensor)
            configs.append(config)
        
        tensors = np.concatenate(tensors, axis=0)

        # model prediction
        if len(tensors.shape) == 4:
            tensors = [tensors] 
        outputs = self.session.run(self.opts, dict(zip(self.inps, tensors)))[0]
        
        outputs = [self.postprocess(output) for output in outputs]

        return outputs, configs if return_dict else outputs
    
    def get_bbox(self, image, bbox, w, h):
        x1, y1, x2, y2 = bbox.astype(int)
        x_min, y_min = max(x1, 0), max(y1, 0)
        x_max, y_max = min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)
        bbox_width, bbox_height = abs(x_max - x_min), abs(y_max - y_min)
        
        x_min = max(0, x_min-int(0.2*bbox_height))
        y_min = max(0, y_min-int(0.2*bbox_width))
        x_max = x_max+int(0.2*bbox_height)
        y_max = y_max+int(0.2*bbox_width)
        
        configs = {
            "x_min" : x_min,
            "y_min" : y_min,
            "x_max" : x_max,
            "y_max" : y_max,
            "bbox_width" : bbox_width,
            "bbox_height" : bbox_height
        }
        
        return configs
    
    def preprocess(self, im:np.array) -> list:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image

        Returns:
            im: image after normalize and resize
        """
        # Normalize the image
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (224, 224))
        im = im / 255.0
        im = (im - self.mean) / self.std
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
    
        return im
    
    def postprocess(self, predict):
        if len(predict.shape) == 2:
            predict = np.expand_dims(predict, 0)
        euler = utils.compute_euler_angles_from_rotation_matrices(
            predict)*180/np.pi
        p_pred = euler[:, 0].detach().cpu().item()
        y_pred = euler[:, 1].detach().cpu().item()
        r_pred = euler[:, 2].detach().cpu().item()

        return p_pred, y_pred, r_pred

if __name__ == '__main__':
    from head import HeadFaceModel
    
    headface = HeadFaceModel('weights/yolov7-headface-v1.onnx')
    headpose = HeadPoseModel('weights/cmu.onnx')
    
    img = cv2.imread('images/snapshot.jpg')
    h, w, _ = img.shape
    bboxes, scores, labels, kpts = headface.inference(img)
    crops = []
    
    results, configs = headpose.predict(img, bboxes[0], return_dict=True)
    print("Single batch: ", configs)
    
    results, configs = headpose.batch_predict(
        imgs = [img, img, img],
        bboxes= [bboxes[0], bboxes[0], bboxes[0]],
        return_dict=True
    )
    print("\n\nDynamic batch: ", configs)
    # cv2.imwrite("full_predict.jpg", result['image'])