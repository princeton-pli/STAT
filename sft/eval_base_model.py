import torch
import json
import re
from tqdm import tqdm
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import os
import sys

import os
import json
from tqdm import tqdm
from transformers import GenerationConfig
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eval')))
from eval import parser
from eval import grader

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a Llama 3 model with SFT and QLoRA.")

    parser.add_argument("--base_model_id", type=str, default="./models/Llama-3.2-3B-Instruct")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft")
    parser.add_argument("--output_filename", type=str, default="")
    parser.add_argument("--base_output_filename", type=str, default="")
    parser.add_argument("--run_base_model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1, help="generation temperature")
    parser.add_argument("--num_generated_responses", type=int, default=1, help="Number of responses to generate per prompt (for pass@k evaluation).")

    return parser.parse_args()

class EvalConfig:
    # Model and tokenizer names
    BASE_MODEL_ID = "./models/Llama-3.2-3B-Instruct"
    # BASE_MODEL_ID = "./models/Qwen2.5-1.5B-Instruct"

    DATA_NAME = "original_train_multi_sol_9.5k"
    ADAPTER_PATH = f"./sft/finetuned_adapters/Llama-3.2-3B-Instruct/sft_{DATA_NAME}_5e-6_Llama-3.2-3B-Instruct/final_adapter"
    # OUTPUT_FILENAME = "eval_results_sft_7.5k_5e-06_Llama-3.2-3B-Instruct_lr5e-06_ep3.jsonl"
    OUTPUT_FILENAME = f"eval_results_math500_sft_{DATA_NAME}_5e-06_llama3.2_3b_instruct.jsonl"
    BASE_OUTPUT_FILENAME = "base_model_results_math.jsonl"
    # BASE_OUTPUT_FILENAME = "base_model_results_math500.jsonl"
    # Test dataset
    # TEST_DATA_PATH = "./sft/eval_data/amc23.jsonl"
    # TEST_DATA_PATH = "/scratch/gpfs/yh0068/slm-math/RLHF-Reward-Modeling/pipeline_outputs/skill_labeled/math/Llama-3.2-3B-Instruct/skills_labeled.jsonl"  
    TEST_DATA_PATH = "/scratch/gpfs/yh0068/slm-math/RLHF-Reward-Modeling/evaluation/data/math/test.jsonl"
    # Output directory for saving results
    OUTPUT_DIR = "./sft/evaluation_results"
    NUM_TEST_SAMPLES = 500
    NUM_GENERATED_RESPONSES = 1

    # Fixed instruction for all test questions
    INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"
    
    # Generation parameters
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.1 # Use low temperature for more deterministic, factual answers
    TOP_P = 0.9
    DO_SAMPLE = True

    print(f"ADAPTER PATH: {ADAPTER_PATH}")

# ---------------------------------------------------------------------------------
# 3. Main Evaluation Logic
# ---------------------------------------------------------------------------------

def format_prompt(tokenizer, instruction, question):
    """
    Formats the prompt for a base model, consistent with fine-tuning.
    This should reflect how you structured your training data (e.g., "Instruction: X\nQuestion: Y\nAnswer:").
    """
    # Assuming your fine-tuning data was structured like:
    # "Instruction: {instruction}\nQuestion: {question}\nAnswer:"
    # This will make the model complete the "Answer:" part.
    # Adjust this string precisely to match your fine-tuning input format.
    prompt_text = f"{instruction}\n{question}" 
    # Or, if your fine-tuning data was just "Instruction here. Question here. Output here."
    # then perhaps: f"{instruction} {question} "
    
    return prompt_text # No chat template for base models


def evaluate_model(model, tokenizer, dataset, config, output_filename=None, overwrite=False):
    """
    Runs evaluation. If results file exists, it re-grades from the file.
    Otherwise, it runs model generation and saves the results.
    """
    # if not output_filename:
    #     output_filename = config.OUTPUT_FILENAME
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)
    
    # --- Check if results file already exists ---
    if os.path.exists(output_path):
        print(f"\n--- Found existing results at {output_path} ---")
        print("--- Re-running answer extraction and grading only ---")

        difficult_question = []
        question_to_answer = {}

        # print(dataset[0])
        for example in dataset:
            # if not example['parsed_output']['judge_pred'] or True not in example['score']:
            question_to_answer[example["question"]] = example["answer"]
            if 'score' in example and True not in example['score']:
                difficult_question.append(example['question'])
        
        results = []
        with open(output_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        correct_predictions = 0
        total_predictions = 0
        for result in tqdm(results, desc="Re-grading"):
            # if difficult_question and result['question'] not in difficult_question:
            #     continue
            # Re-run the answer extraction on the stored model output
            if "gsm8k" in config.TEST_DATA_PATH:
                question = result['question']
                solution = question_to_answer[question]
                ground_truth_answer = str(parser.extract_answer(solution, 'gsm8k'))
                result['ground_truth'] = ground_truth_answer
                extracted_answer = parser.extract_answer(result['full_model_output'], 'gsm8k')
            else:
                extracted_answer = parser.extract_answer(result['full_model_output'], 'math')
            
            # Compare with the ground truth using the grader
            if "boxed" in result['ground_truth']:
                    result['ground_truth'] = str(parser.extract_answer(result['ground_truth'], 'math'))
            is_correct = grader.math_equal(extracted_answer, result['ground_truth'])
            
            # Update the result dictionary and count correct predictions
            result['extracted_answer'] = extracted_answer
            result['is_correct'] = is_correct
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
        
        # Calculate and print the new accuracy
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Re-grading finished. New Accuracy: {accuracy:.2f}%")
        
        # Overwrite the old file with the re-graded results
        if overwrite:
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print("Updated results file with new grading.")
        
        return accuracy

    # --- If file does not exist, run the full generation and evaluation ---
    else:
        print(f"\n--- Starting new evaluation for: {output_filename} ---")
        
        results = []
        correct_predictions = 0
        pass_k_correct_questions = 0
        total_correct_responses = 0
        total_responses_generated = 0
        
        generation_config = GenerationConfig(
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=config.DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=config.NUM_GENERATED_RESPONSES,
        )
        
        for example in tqdm(dataset, desc="Evaluating"):
            if "question" in example:
                question = example["question"]
            elif "problem" in example:
                question = example["problem"]
            
            if "gsm8k" in config.DATA_NAME:
                ground_truth_answer = str(parser.extract_answer(example["answer"], 'gsm8k'))
            elif "gt" in example:
                ground_truth_answer = str(example["gt"])
            elif "answer" in example:
                ground_truth_answer = str(example["answer"])
            prompt = format_prompt(tokenizer, config.INSTRUCTION, question)
            print("Prompt: ", prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)
            
            # full_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # model_output = full_prediction[len(prompt.replace(tokenizer.bos_token, '')):]
            if config.NUM_GENERATED_RESPONSES == 1:
                input_token_len = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_token_len:]
                model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                print("Model output: ", model_output)
                
                extracted_answer = parser.extract_answer(model_output, 'math')
                
                # Note: Using math_equal directly, as requested in your original script.
                # If you have a 'grader' object, you'd use grader.math_equal here.

                if "boxed" in ground_truth_answer:
                    ground_truth_answer = str(parser.extract_answer(ground_truth_answer, 'math'))
                is_correct = grader.math_equal(extracted_answer, ground_truth_answer) 

                print("Is_correct: ", is_correct)
                if not is_correct:
                    print(f"correct answer: {ground_truth_answer}, model answer: {extracted_answer}")
                if is_correct:
                    correct_predictions += 1
                    
                results.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth": ground_truth_answer,
                    "full_model_output": model_output,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct
                })
            else:
                input_token_len = inputs.input_ids.shape[1]
                model_outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)

                generated_responses_list = []
                question_is_correct_for_pass_k = False

                for model_output in model_outputs:
                    total_responses_generated += 1
                    extracted_answer = parser.extract_answer(model_output, 'math')
                    is_correct = grader.math_equal(extracted_answer, ground_truth_answer)
                    
                    if is_correct:
                        total_correct_responses += 1
                        question_is_correct_for_pass_k = True
                    
                    generated_responses_list.append({
                        "full_model_output": model_output,
                        "extracted_answer": extracted_answer,
                        "is_correct": is_correct
                    })
                
                if question_is_correct_for_pass_k:
                    pass_k_correct_questions += 1
                    
                results.append({
                    "question": question,
                    "prompt": prompt,
                    "ground_truth": ground_truth_answer,
                    "generated_responses": generated_responses_list,
                    "is_correct_any": question_is_correct_for_pass_k
                })
            
        accuracy = (correct_predictions / len(dataset)) * 100
        print(f"Evaluation finished. Accuracy: {accuracy:.2f}%")
        
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"Results saved to {output_path}")
        
        return accuracy

def main():
    args = get_args()
    config = EvalConfig()

    config.BASE_MODEL_ID = args.base_model_id
    config.DATA_NAME = args.data_name
    config.ADAPTER_PATH = args.adapter_path
    config.TEST_DATA_PATH = args.test_data_path

    config.OUTPUT_DIR = args.output_dir
    config.OUTPUT_FILENAME = args.output_filename
    config.BASE_OUTPUT_FILENAME = args.base_output_filename

    config.NUM_TEST_SAMPLES = args.num_test_samples
    config.NUM_GENERATED_RESPONSES = args.num_generated_responses
    config.TEMPERATURE = args.temperature
    
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load the test dataset
    test_dataset = load_dataset("json", data_files=config.TEST_DATA_PATH, split="train")
    if config.NUM_TEST_SAMPLES != -1:
        test_dataset = test_dataset.select(range(min(len(test_dataset), config.NUM_TEST_SAMPLES)))

    # Load tokenizer (same for both models)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. Evaluate the Base Model ---
    output_path_base = os.path.join(config.OUTPUT_DIR, config.BASE_OUTPUT_FILENAME)
    output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
    
    if (args.run_base_model and not os.path.exists(output_path_base)) or (not args.run_base_model and not os.path.exists(output_path)):
        print("\nLoading Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("\nLoading Fine-Tuned Model for evaluation...")
        # Load the PEFT model by applying the adapter to the base model
        if os.path.exists(config.ADAPTER_PATH):
            ft_model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
            merged_model = ft_model.merge_and_unload()
        else:
            merged_model = None
    else:
        base_model = None
        merged_model = None

    if args.run_base_model:
        evaluate_model(base_model, tokenizer, test_dataset, config, config.BASE_OUTPUT_FILENAME, args.overwrite)
    
    # --- 2. Evaluate the Fine-Tuned Model ---
    
    # if merged_model:
    else:
        evaluate_model(merged_model, tokenizer, test_dataset, config, config.OUTPUT_FILENAME, args.overwrite)

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
