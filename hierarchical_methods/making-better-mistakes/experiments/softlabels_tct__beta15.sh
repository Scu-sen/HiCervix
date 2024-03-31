OMP_NUM_THREADS=2 MKL_NUM_THREADS=2  python3 ../scripts/start_training.py \
    --arch resnet50 \
    --loss soft-labels \
    --lr 1e-5 \
    --data inaturalist19-224 \
    --beta 15 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output softlabels_tct_beta15/ \
    --num_training_steps 200000

