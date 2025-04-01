import os
import onnx
import torch
import argparse
from model import RepNet6D

import importlib.util


def export(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    model = RepNet6D(backbone_name='RepVGG-B1g4',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)
    # Load snapshot
    saved_state_dict = torch.load(os.path.join(args.snapshot_model), map_location=None if torch.cuda.is_available() else 'cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    
    model.to(device)
    model.eval()
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,224,224) iDetection
    opt_tensor = model(img)
    print("opt_tensor: ", opt_tensor.shape)
    
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = args.snapshot_model.replace('.pth', '.onnx')  # filename
    output_names = ['output'] # output name
    
    if args.dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
            'output': {0: 'batch', 2: 'y', 3: 'x'}}
    if args.dynamic_batch:
        args.batch_size = 'batch'
        dynamic_axes = {
            'images': {0: 'batch'}, 
            'output': {0: 'batch'},
        }

    torch.onnx.export(
        model, 
        img, 
        f, 
        verbose=False, 
        opset_version=args.opset, 
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    # Simplify
    if args.simplify:
        try:
            cuda = torch.cuda.is_available()
            # Check if package is installed
            if importlib.util.find_spec("onnx-simplifier") is None:
                if cuda:
                    os.system("pip install onnxruntime-gpu")
                else:
                    os.system("pip install onnxruntime onnx-simplifier>=0.4.1")
            import onnxsim

            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f'Simplifier failure: {e}')
    
    # cleanup
    if args.cleanup:
        try:
            # Check if package is installed
            if importlib.util.find_spec("onnx_graphsurgeon") is None:
                os.system("pip install onnx_graphsurgeon")

            import onnx_graphsurgeon as gs

            print("Starting to cleanup ONNX using onnx_graphsurgeon...")
            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    inputs = [node for node in model_onnx.graph.input]
    print("Inputs: ", inputs, "\n")
    outputs = [node for node in model_onnx.graph.output]
    print("Outputs: ", outputs)
    
    return f, model_onnx



if __name__ == '__main__':
    
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--snapshot-model', help='Name of model snapshot.',
                        default='./cmu.pth', type=str)
    parser.add_argument('--batch-size', help='Number of image batch',
                        default=1, type=int)
    parser.add_argument('--img-size', nargs='+', type=int, 
                        default=[224, 224], help='image size')  # height, width
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--cleanup', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')

    args = parser.parse_args()
    export(args)
    

