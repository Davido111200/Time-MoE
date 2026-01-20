
model_name=RAFTTimeMoE
pipeline=RAFTTimeMoE
seq_len=512
pred_lens=(96 192 336 720)
top_k=(20)
for pred_len in ${pred_lens[@]}
do
  for k in ${top_k[@]}
  do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./data/all_datasets/weather/ \
    --data_path weather.csv \
    --model_id weather_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len ${seq_len} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --n_period 3 \
    --topm ${k} \
    --learning_rate 0.001 \
    --model_path Time_MoE/checkpoints/50M_weather/ 

  done
done
