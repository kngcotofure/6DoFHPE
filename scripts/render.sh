CUDA_VISIBLE_DEVICES=0 python demo/demo_3DoF.py \
    --source 'videos' \
    --snapshot-model 'weights/cmu.pth' \
    --head-model 'weights/Head_detection_300x300.pb' \
    --device '0'