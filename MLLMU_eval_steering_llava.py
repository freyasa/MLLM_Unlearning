import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
import pickle
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration
import pandas as pd
from io import BytesIO
# from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration, MllamaProcessor, MllamaForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import LlavaConfig
import argparse
import fnmatch
import string
from refusal_direction.qwen_steering import AlphaQwen2_5_VLDecoderLayer, Qwen2_5_VLModel_for_Steering, QwenForSteering
from newcode.llava_steering import LlamaModel_for_Steering, LlavaForSteering
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
import logging

few_shots_num=0
def load_and_combine_parquet_files(directory):
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

def compute_bleu(ground_truth, predicted_answer):
    reference = [ground_truth.split()]  
    hypothesis = predicted_answer.split() 
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    return bleu_score

def formulate_prompt_with_options(question, options):
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt

def evaluate_classification(parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None, logger=None):
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    logger.info(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for j, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        classification_questions = row["Classification_Task"]
        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Iterate through each image-textual question
        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            if "llama" in args.model_id.lower():
                prompt = (f"USER: <|image|><|begin_of_text|>\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:")
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            elif "llava" in args.model_id.lower():
                prompt = (f"USER: <image>\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:")
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            elif "qwen" in args.model_id.lower():
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question_with_options}\nJust give ONE letter representing the answer directly."},
                            ],
                        }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                prompt = prompt.strip()
                inputs = processor(images=image,text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
            generated_text=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
            assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    logger.info(f"Classification image-textual Question Accuracy: {image_textual_accuracy:.2f}%")

    return {
        "Classification image-textual Question Accuracy": image_textual_accuracy
    }


def evaluate_fill_in_the_blank(parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None, logger=None):
    logger.info("################################## Fill-in-the-blank Task Starts ##############################################")
    logger.info(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    logger.info(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for j, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        fill_in_the_blank_questions = row["Mask_Task"]
        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each question in Mask_Task and skip if it's a few-shot question
        for idx, question_entry in enumerate(fill_in_the_blank_questions):
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

            # Model specific logic
            if "llama" in args.model_id.lower():
                prompt = (f"USER: "
                    f"<|image|><|begin_of_text|>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                    f"USER:\n{question}\nASSISTANT:")
                inputs = processor(images=image if question_type == "Image_Textual" else None,
                                text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            elif "llava" in args.model_id.lower():
                prompt = (f"USER: "
                    f"<image>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                    f"USER: {question}\nASSISTANT:")
                if question_type == "Image_Textual":
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            elif "qwen" in args.model_id.lower():
                if question_type == "Image_Textual":
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}"},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                prompt = prompt.strip()
                if question_type == "Image_Textual":
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                else:   
                    inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
            assistant_response=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)

            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    logger.info(f"Cloze Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")

    return {
        "Cloze image_textual_accuracy": image_textual_accuracy
    }

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None, logger=None):
    logger.info("################################## Generation Task Starts ##############################################")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Loop through each person's data in the evaluation samples
    for j, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]
        # Randomly select one image if in test mode and multiple images are available
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Process each generation question
        for question_data in generation_questions:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                if "llava" in args.model_id.lower():
                    prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif "qwen" in args.model_id.lower():
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."},
                            ],
                        }
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    prompt = prompt.strip()
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif "llama" in args.model_id.lower():
                    prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                else:
                    raise ValueError("Model ID not supported")
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
                predicted_answer=tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1

    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    for metric, score in avg_scores.items():
        logger.info(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")
    parser.add_argument('--model_id', type=str, required=False, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--cache_path', type=str, required=False)
    parser.add_argument('--data_split_folder', type=str, required=False)
    parser.add_argument('--few_shot_data', type=str, required=False)
    parser.add_argument('--test_data', type=str, required=False)
    parser.add_argument('--celebrity_data', type=str, required=False)
    parser.add_argument('--forget_ratio', type=int, default=10)
    parser.add_argument('--eval_list', type=str, required=False, default='classification_generation_cloze')
    parser.add_argument('--task', type=str, required=False, default='image')
    parser.add_argument('--steering_matrix_path', type=str, required=False)

    parser.add_argument('--cloze_strength', type=float, required=False, default=-0.25)
    parser.add_argument('--generation_strength', type=float, required=False, default=-0.35)
    parser.add_argument('--classification_strength', type=float, required=False, default=-0.25)


    return parser.parse_args()

def main():
    args = parse_arguments()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'MLLMU-Bench')
    args.test_data = os.path.join(data_dir, 'Test_Set')
    args.celebrity_data = os.path.join(data_dir, 'Retain_Set')
    args.data_split_folder = data_dir
    args.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLaVA_Vanilla')

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),"log_llava"),exist_ok=True)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"log_llava")

    if args.eval_list == "classification_generation_cloze":
        log_file = f"{log_path}/all_cloze_{args.cloze_strength}_generation_{args.generation_strength}_classification_{args.classification_strength}_{args.forget_ratio}.log"
    elif args.eval_list == "classification":
        log_file = f"{log_path}/{args.steering_matrix_path_str}_classification_{args.classification_strength}_{args.forget_ratio}.log"
    elif args.eval_list == "generation":
        log_file = f"{log_path}/{args.steering_matrix_path_str}_generation_{args.generation_strength}_{args.forget_ratio}.log"
    elif args.eval_list == "cloze":
        log_file = f"{log_path}/{args.steering_matrix_path_str}_cloze_{args.cloze_strength}_{args.forget_ratio}.log"
    
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_file)
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    logger.info(f"Forget Folder: {forget_folder}")
    logger.info(f"Retain Folder: {retain_folder}")
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # real_paraquet_file = os.path.join(args.celebrity_data, f"train-00000-of-00001.parquet")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if "qwen" in args.model_id.lower():
        tokenizer.padding_side = "left"
    torch.cuda.empty_cache()

    if "llava" in args.model_id.lower():
        logger.info("Loading LLAVA Vanilla model...")
        model = LlavaForSteering.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        processor.tokenizer.padding_side = "right"
        config = LlavaConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
        hidden_dim = config.text_config.hidden_size
        num_layers = config.text_config.num_hidden_layers
        
        steering_matrix = torch.load(args.steering_matrix_path, map_location=torch.device("cuda"))
        steering_matrix = steering_matrix.to(torch.float16)
        steering_matrix = steering_matrix.to(model.device)
    elif "llama" in args.model_id.lower():
        logger.info("Loading idefics2 Vanilla model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )

    strength = [0.0] * num_layers
    new_steering_layers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    if "classification" in args.eval_list:

        for layer in new_steering_layers:
            strength[layer] = args.classification_strength
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)

        logger.info("### Evaluating classification task ###")
        forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget",
            logger=logger)
        test_classification_result = evaluate_classification(parquet_file=args.test_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="test",
                                                                    forget_parquet_file=forget_parquet_file,
                                                                    logger=logger)

        retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="retain_shared",
                                                            logger=logger)

        real_classification_result = evaluate_classification(parquet_file=args.celebrity_data,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="retain_celebrity",
                                                            logger=logger)
        

        logger.info("Classification Results:")
        logger.info(f"forget_classification_result:{forget_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"test_classification_result:{test_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"retain_classification_result:{retain_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"real_classification_result:{real_classification_result['Classification image-textual Question Accuracy']}")

    if "generation" in args.eval_list:

        for layer in new_steering_layers:
            strength[layer] = args.generation_strength
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)

        logger.info("### Evaluating cloze task ###")
        forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                            processor=processor,
                                                            tokenizer=tokenizer,
                                                            model=model,
                                                            args=args,
                                                            mode="forget",
                                                            logger=logger
                                                            )
        test_generation_result = evaluate_generation(parquet_file=args.test_data,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="test",
                                                    forget_parquet_file=forget_parquet_file,
                                                    logger=logger)

        retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="retain_shared",
                                                    logger=logger)
        real_generation_result = evaluate_generation(parquet_file=args.celebrity_data,
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    model=model,
                                                    args=args,
                                                    mode="retain_celebrity",
                                                    logger=logger)

        logger.info("Generation Results Textual:")
        logger.info(f"forget_generation_result:{forget_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"test_generation_result:{test_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"retain_generation_result:{retain_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"real_generation_result:{real_generation_result['Average ROUGE-L (Image_Textual)']}")
        
    if "cloze" in args.eval_list:

        for layer in new_steering_layers:
            strength[layer] = args.cloze_strength
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)

        logger.info("### Evaluating cloze task ###")
        forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget",
            logger=logger)

        test_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.test_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="test",
                                                                    forget_parquet_file=forget_parquet_file,
                                                                    logger=logger)

        retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="retain_shared",
                                                                    logger=logger)
        
        real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.celebrity_data,
                                                                    processor=processor,
                                                                    tokenizer=tokenizer,
                                                                    model=model,
                                                                    args=args,
                                                                    mode="retain_celebrity",
                                                                    logger=logger)

        logger.info("Cloze Results:")
        logger.info(f"forget_fill_in_the_blank_result:{forget_fill_in_the_blank_result['Cloze image_textual_accuracy']}")
        logger.info(f"test_fill_in_the_blank_result:{test_fill_in_the_blank_result['Cloze image_textual_accuracy']    }")
        logger.info(f"retain_fill_in_the_blank_result:{retain_fill_in_the_blank_result['Cloze image_textual_accuracy']}")
        logger.info(f"real_fill_in_the_blank_result:{real_fill_in_the_blank_result['Cloze image_textual_accuracy']}")

    if "classification_generation_cloze" in args.eval_list:
        logger.info("************************IMAGE_TEXTUAL RESULT*****************:")
        logger.info("#######################CLASSIFICATION RESULTS################:")
        logger.info(f"Forget Classification image-textual Question Accuracy:{forget_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"Test Classification image-textual Question Accuracy:{test_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"Retain Classification image-textual Question Accuracy:{retain_classification_result['Classification image-textual Question Accuracy']}")
        logger.info(f"Real Classification image-textual Question Accuracy:{real_classification_result['Classification image-textual Question Accuracy']}")

        logger.info("#######################GENERATION RESULTS########################:")
        logger.info(f"Forget Average ROUGE-L (Image_Textual):{forget_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"Test Average ROUGE-L (Image_Textual):{test_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"Retain Average ROUGE-L (Image_Textual):{retain_generation_result['Average ROUGE-L (Image_Textual)']}")
        logger.info(f"Real Average ROUGE-L (Image_Textual):{real_generation_result['Average ROUGE-L (Image_Textual)']}")

        logger.info("#######################CLOZE RESULTS########################:")
        logger.info(f"Forget Cloze image_textual_accuracy:{forget_fill_in_the_blank_result['Cloze image_textual_accuracy']}")
        logger.info(f"Test Cloze image_textual_accuracy:{test_fill_in_the_blank_result['Cloze image_textual_accuracy']}")
        logger.info(f"Retain Cloze image_textual_accuracy:{retain_fill_in_the_blank_result['Cloze image_textual_accuracy']}")
        logger.info(f"Real Cloze image_textual_accuracy:{real_fill_in_the_blank_result['Cloze image_textual_accuracy']}")

if __name__ == "__main__":
    main()