import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model with SFT and QLoRA.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the existing LoRA adapter to continue training.")
    parser.add_argument("--output_dir", type=str, default="./sft_results", help="Base directory to save the fine-tuned model and checkpoints.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The initial learning rate for the AdamW optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum total input sequence length after tokenization.")
    return parser.parse_args()

# ---------------------------------------------------------------------------------
# Data Formatting
# ---------------------------------------------------------------------------------
def format_dataset_to_text(examples):
    """
    Formats a dataset with 'question', 'solution' columns into raw text
    for base-model style SFT (no chat template). The model learns to
    continue with the solution after "{instruction}\n{question}".
    """
    formatted_texts = []
    for i in range(len(examples['question'])):
        instruction = "Please reason step by step, and put your final answer within \\boxed{}"
        input_text = examples['question'][i]
        output_text = examples['solution'][i]

        # Concatenate instruction, input, and output directly for a base model
        text = f"{instruction}\n{input_text}\n{output_text}"
        formatted_texts.append(text.strip())
    return {"text": formatted_texts}

def main():
    args = get_args()

    # --- W&B and Output Directory Setup ---
    model_name_safe = args.model_id.split('/')[-1]
    run_name = f"sft_{os.path.basename(args.dataset_path).split('.')[0]}_{model_name_safe}_lr{args.learning_rate}_ep{args.num_train_epochs}"
    output_dir = os.path.join(args.output_dir)
    
    print(f"RUN NAME: {run_name}")
    print(f"OUTPUT DIR: {output_dir}")

    # ---------------------------------------------------------------------------------
    # 1. Configuration (from arguments)
    # ---------------------------------------------------------------------------------
    
    # QLoRA configuration
    USE_QLORA = True
    LORA_R = 16 # LoRA attention dimension (rank)
    LORA_ALPHA = 32 # Alpha parameter for scaling
    LORA_DROPOUT = 0.05 # Dropout probability for LoRA layers
    # Target modules for Qwen2. To find these, print `model.named_modules()`
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # bitsandbytes configuration
    BNB_4BIT_QUANT_TYPE = "nf4" # "nf4" or "fp4"
    BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
    USE_NESTED_QUANT = False

    # Training arguments
    OPTIM = "paged_adamw_8bit"
    SAVE_STEPS = 1000
    LOGGING_STEPS = 10
    PACKING = False # Packing can increase efficiency for datasets with many short examples

    # ---------------------------------------------------------------------------------
    # 2. Data Loading and Formatting
    # ---------------------------------------------------------------------------------
    
    # Load the dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    
    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Apply formatting for base-model style raw-text SFT
    final_dataset = dataset.map(format_dataset_to_text, batched=True, remove_columns=dataset.column_names)

    print("\n--- Sample of final formatted dataset ---")
    print(final_dataset[0]['text'])
    print("------------------------------------------\n")

    # ---------------------------------------------------------------------------------
    # 3. Model Loading and Configuration
    # ---------------------------------------------------------------------------------
    
    # Configure quantization
    if USE_QLORA:
        compute_dtype = BNB_4BIT_COMPUTE_DTYPE
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        )
    else:
        bnb_config = None

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype if USE_QLORA else torch.float16,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Configure LoRA for training
    if USE_QLORA:
        if args.adapter_path:
            print(f"Loading existing adapter from {args.adapter_path} to continue training...")
            # Load the existing adapter and set it to trainable
            model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
            peft_config = None # Trainer will use the loaded adapter's config
        else:
            print("No adapter path provided. Creating a new LoRA adapter...")
            # Create a new adapter config
            peft_config = LoraConfig(
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                r=LORA_R,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=LORA_TARGET_MODULES,
            )
            model = get_peft_model(base_model, peft_config)
        
        model.print_trainable_parameters()
    else:
        # If not using QLoRA/PEFT, the model is just the base model
        model = base_model
        peft_config = None
    
    # model.to('cuda')

    # ---------------------------------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------------------------------

    # Configure training arguments
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=OPTIM,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        fp16=not USE_QLORA,
        bf16=USE_QLORA,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        weight_decay=0.1,
        group_by_length=True,
        report_to="none",
        run_name=run_name,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=PACKING,
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=final_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the final adapter
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    print(f"Adapter saved to {final_adapter_dir}")
    
    # Finish the W&B run
    wandb.finish()

    # ---------------------------------------------------------------------------------
    # 5. Inference with the fine-tuned model (sanity check)
    # ---------------------------------------------------------------------------------
    print("\n--- Testing the fine-tuned model ---")
    
    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the PEFT model with the saved adapter
    model = PeftModel.from_pretrained(base_model, final_adapter_dir)
    
    # Merge the adapter into the base model for faster inference
    model = model.merge_and_unload()

    # Create a text generation pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Example prompt for a base model.
    # The format here should ideally match the format you used during fine-tuning (e.g., "Prompt: X\nResponse: ").
    # If your fine-tuning data was just raw text, then a simple prompt will lead to continuation.
    prompt_for_inference = "Generate a short story about a brave knight and a wise dragon guarding an ancient artifact."
    formatted_prompt = f"Story start: {prompt_for_inference}\n" # Example: matching a potential training format
    
    sequences = text_generator(
        formatted_prompt,
        max_new_tokens=250, # Increased for potentially longer generation from base models
        do_sample=True,
        temperature=0.7,   # Added temperature for more creative output
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"Input Prompt for Inference: {formatted_prompt}")
    # For base models, the output typically starts right after the input.
    # You might need to manually parse or split the generated text.
    full_generated_text = sequences[0]['generated_text']
    # Attempt to extract only the generated part after the input prompt
    if full_generated_text.startswith(formatted_prompt):
        assistant_response = full_generated_text[len(formatted_prompt):].strip()
    else:
        # Fallback if prefix matching is not perfect (e.g., due to tokenization oddities)
        assistant_response = full_generated_text.strip()
        print("Warning: Could not perfectly strip prompt from generated text.")

    print(f"\nGenerated response:\n{assistant_response}")

if __name__ == "__main__":
    main()