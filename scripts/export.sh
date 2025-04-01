python export_onnx.py \
    --snapshot-model 'weights/cmu.pth' \
    --batch-size 10 \
    --dynamic-batch \
    --simplify \
    --cleanup \
    --opset 12