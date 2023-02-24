CUDA_VISIBLE_DEVICES=3 python mst/scripts/train.py \
--batch_size 8 \
--num_epochs 400 \
--lr 1e-4 \
--num_workers 8 \
--loss_criteria STFTloss \
--dataset MedleyDB \
--model encoder_with_resnet \
--experiment_type gain_pan \
--num_tracks [4,25] \
--diff_sections True \
--sample_rate 44100 \
--duration 1 \
--log_dir ./logs \ 





# Path: mix_style_transfer/mst/configs/train_medleydb_resnet.sh


