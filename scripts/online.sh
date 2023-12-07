
CUDA_VISIBLE_DEVICES=4 python scripts/online.py \
--track_dir "/import/c4dm-datasets-ext/test-multitracks/Kat Wright_By My Side" \
--ref_mix "/import/c4dm-datasets-ext/diffmst_validation/listening/diffmst-examples_wavref/The Dip - Paddle To The Stars (Lyric Video).wav" \
--use_gpu \
--n_iters 1000 \
--loss "feat" \
#--stem_separation \