# Extract the forget and retain data embeddings
CUDA_VISIBLE_DEVICES=0 python ./newcode/embed_qwen.py --dataset forget --forget_ratio 10
CUDA_VISIBLE_DEVICES=0 python ./newcode/embed_qwen.py --dataset retain --forget_ratio 10

# Calculate the steering matrix
CUDA_VISIBLE_DEVICES=0 python ./newcode/all_calculate_steering_matrix.py --model qwen \
                --this_ratio 0.4 --lambda_reg_para 1.0 --forget_ratio 10 \

# Evaluate the steering 
CUDA_VISIBLE_DEVICES=0 python MLLMU_eval_steering_qwen.py --forget_ratio 10 \
        --steering_matrix_path ./data/MLLMU-Bench/steering_matrix/qwen_0.4_1.0_10.pt 

# Evaluate the Vanilla model
CUDA_VISIBLE_DEVICES=0 python MLLMU_eval.py  --model_id Qwen/Qwen2.5-VL-7B-Instruct --forget_ratio 10\
    --cache_path ./Qwen_Vanilla > Qwen_Vanilla.log

# Train baseline and evaluate
CUDA_VISIBLE_DEVICES=0,1 python ./baselines/MLLMU_NPO.py --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --vanilla_dir ./Qwen_Vanilla --oracle_model_id ./Qwen_Vanilla \
    --save_dir ./NPO_qwen --data_split_dir ./baseline_train_split \
    --batch_size 1 --beta 0.4 --lr 2e-5 --forget_split_ratio 10 --num_epochs 2

CUDA_VISIBLE_DEVICES=0 python MLLMU_eval.py  --model_id Qwen/Qwen2.5-VL-7B-Instruct --forget_ratio 10 \
    --cache_path ./NPO_qwen > NPO_qwen_eval.log