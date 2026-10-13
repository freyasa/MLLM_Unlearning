# Extract the refusal direction
CUDA_VISIBLE_DEVICES=2 python ./refusal_direction/extract_llava.py

# Extract the forget and retain data embeddings
CUDA_VISIBLE_DEVICES=2 python ./newcode/embed_llava.py --dataset forget --forget_ratio 10
CUDA_VISIBLE_DEVICES=2 python ./newcode/embed_llava.py --dataset retain --forget_ratio 10

# Calculate the steering matrix
CUDA_VISIBLE_DEVICES=2 python ./newcode/all_calculate_steering_matrix.py --model qwen \
                --this_ratio 0.3 --lambda_reg_para 0.1 --forget_ratio 10 \

# Evaluate the steering matrix
CUDA_VISIBLE_DEVICES=2 python MLLMU_eval_steering_llava.py --forget_ratio 10 \
        --steering_matrix_path ./data/MLLMU-Bench/steering_matrix/llava_0.3_0.1_10.pt 

# Evaluate the Vanilla model
CUDA_VISIBLE_DEVICES=2 python MLLMU_eval.py  --model_id llava-hf/llava-1.5-7b-hf --forget_ratio 10\
    --cache_path ./LLaVA_Vanilla > LLaVA_Vanilla.log

# Train baseline and evaluate
CUDA_VISIBLE_DEVICES=2,3 python ./baselines/MLLMU_NPO.py --model_id llava-hf/llava-1.5-7b-hf \
    --vanilla_dir ./LLaVA_Vanilla --oracle_model_id ./LLaVA_Vanilla \
    --save_dir ./NPO_llava --data_split_dir ./baseline_train_split \
    --batch_size 4 --beta 0.9 --lr 1e-5 --forget_split_ratio 10 --num_epochs 2

CUDA_VISIBLE_DEVICES=2 python MLLMU_eval.py  --model_id llava-hf/llava-1.5-7b-hf --forget_ratio 10 \
    --cache_path ./NPO_llava > NPO_llava_eval.log