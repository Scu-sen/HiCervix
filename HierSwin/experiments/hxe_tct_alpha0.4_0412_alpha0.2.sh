OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/start_training.py \
    --arch swinT \
    --loss hierarchical-cross-entropy \
    --alpha 0.2 \
    --dropout 0.5 \
    --data inaturalist19-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output hxe_tct_alpha0.4_0412_alpha0.2/ \
    --num_training_steps 30000

