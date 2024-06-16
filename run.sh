python stage1_run.py --config config/vox-256-emo.yaml --device_ids 0,1,2,3

python stage2_run.py --config config/Mix-emo.yaml --device_ids 0,1,2,3 \
    --checkpoint "pretrain_checkpoint_path"
