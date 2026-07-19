python train.py \
  -b 1 \
  -e 5 \
  -l 0.0003 \
  -r 10 \
  -n 'test_PromptMR' \
  -t '/root/Data/train/' \
  -v '/root/Data/val/' \
  --model promptmr \
  --cascade 4 \
  --num_adj_slices 5 \
  --n_feat0 24 \
  --feature_dim 36 48 60 \
  --prompt_dim 12 24 36 \
  --sens_n_feat0 12 \
  --sens_feature_dim 18 24 30 \
  --sens_prompt_dim 6 12 18 \
  --n_history 3 \
  --n_buffer 4
  # add --use_checkpoint --compute_sens_per_coil if VRAM is short
  # paper-size config: --cascade 12 --n_feat0 48 --feature_dim 72 96 120 \
  #   --prompt_dim 24 48 72 --sens_n_feat0 24 --sens_feature_dim 36 48 60 \
  #   --sens_prompt_dim 12 24 36 --n_history 11
