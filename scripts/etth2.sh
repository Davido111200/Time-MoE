
data=ETTh2
python main.py \
    --data_path /scratch/s225250685/project/ICL-time-series/Time-MoE/data/all_datasets/ETT-small/${data}.jsonl \
    --model_path /scratch/s225250685/project/Huggingface/Maple728/TimeMoE-50M \
    --output_path logs/${data}

python run_eval.py -d data/all_datasets/ETT-small/${data}.csv -p 96 -m  logs/${data}
python run_eval.py -d data/all_datasets/ETT-small/${data}.csv -p 192 -m  logs/${data}
python run_eval.py -d data/all_datasets/ETT-small/${data}.csv -p 336 -m  logs/${data}
python run_eval.py -d data/all_datasets/ETT-small/${data}.csv -p 720 -m  logs/${data}