
model_name=RAFTTimerXL
pipeline=RAFTTimerXL
seq_len=96
pred_lens=(24 36 48 60)
top_k=(20)
for pred_len in ${pred_lens[@]}
do
  for k in ${top_k[@]}
  do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/all_datasets/illness/ \
    --data_path national_illness.csv \
    --model_id illness_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --n_period 3 \
    --topm ${k} \
    --learning_rate 0.001 \
    --model_path Time_MoE/checkpoints/timer_finetuned_all_channels/illness/epoch-1

  done
done
