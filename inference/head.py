import cv2
import numpy as np
import onnxruntime as ort


class HeadFaceModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]) -> None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options, providers)

        self.inps = [inp.name for inp in self.session.get_inputs()]
        self.opts = [opt.name for opt in self.session.get_outputs()]
        _, _, h, w = self.session.get_inputs()[0].shape
        self.model_inpsize = (h, w)
 
    def predict(self, img: np.array, det_thres=0.6, get_layer='head') -> list:
        # preprocess
        tensor, ratio, dwdh = self.preprocess(img, self.model_inpsize)
        if len(tensor.shape) == 4:
            tensor = [tensor]
  
        # prediction
        outputs = self.session.run(self.opts, dict(zip(self.inps, tensor)))
        pred = outputs[1] if get_layer == 'face' else outputs[0]
        
        # postprocess
        result = self.postprocess(pred, ratio, dwdh, det_thres)
        
        return result
    
    def batch_predict(self, imgs: np.array, det_thres=0.6, get_layer='head') -> list:
        # preprocess
        tensors, ratios, dwdhs = [], [], []
        for idx in range(len(imgs)):
            tensor, ratio, dwdh = self.preprocess(imgs[idx], self.model_inpsize)
            tensors.append(tensor)
            ratios.append(ratio)
            dwdhs.append(dwdh)
            
        tensors = np.concatenate(tensors, axis=0)
        if len(tensors.shape) == 4:
            tensors = [tensors]
  
        # prediction
        outputs = self.session.run(self.opts, dict(zip(self.inps, tensors)))
        preds = outputs[1] if get_layer == 'face' else outputs[0]
        preds = np.array(preds)

        # postprocess
        results = []
        for idx in range(len(imgs)):
            tmp = preds[preds[:, 0] == idx]
            results.append(self.postprocess(
                tmp,
                ratios[idx],
                dwdhs[idx],
                det_thres
            ))

        return results
        
    def preprocess(self, im:np.array, new_shape=(640, 640), color=(114, 114, 114), scaleup=True) -> list:
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color
                                )  # add border
        
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255
        
        return im, r, (dw, dh)
    
    def postprocess(self, pred, ratio, dwdh, det_thres = 0.7):
        if isinstance(pred, list):
            pred = np.array(pred)
        
        if len(pred.shape) == 1:
            pred = np.expand_dims(pred, axis=0)
        
        pred = pred[pred[:, 6] > det_thres] # get sample higher than threshold
        
        padding = dwdh*2
        det_bboxes, det_scores, det_labels  = pred[:,1:5], pred[:,6], pred[:, 5]
        kpts = pred[:, 7:] if pred.shape[1] > 6 else None
        det_bboxes = (det_bboxes[:, 0::] - np.array(padding)) / ratio
        
        if kpts is not None:
            kpts[:,0::3] = (kpts[:,0::3] - np.array(padding[0])) / ratio
            kpts[:,1::3] = (kpts[:,1::3]- np.array(padding[1])) / ratio

        return det_bboxes, det_scores, det_labels, kpts
    

if __name__ == '__main__':
    import glob
    
    
    model = HeadFaceModel(
        'weights/yolov7-headface-v1.onnx'
    )
    imgs = [cv2.imread(img) for img in glob.glob("images/*")]
    
    # result = model.predict(img)
    # print("\nSingle batch: ", result)
    
    results = model.batch_predict(imgs)
        
    counts, bboxes = [], []
    for row in results:
        counts.append(len(row[0]))
        bboxes.extend(row[0])
    
    _frame = []
    for idx in range(len(imgs)):
        _frame.extend([imgs[idx]] * counts[idx])
        
    print("\nCounts: ", counts)
    print("\nbboxes: ", bboxes)
    print("\nimgs: ", len(_frame))