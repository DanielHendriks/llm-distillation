import argparse, torch, wandb
from transformers import TrainingArguments
from dataloaders.cqa import CQADatasetLoader
from dataloaders.esnli import ESNLIDatasetLoader
from dataloaders.strategyqa import StrategyQADatasetLoader
from training.custom_metrics import get_compute_metrics_function
from training.decoder_only.datacollators import MultitaskDataCollator, CounterfactualDataCollator, CombinedDataCollator
from training.decoder_only.trainers import MultitaskTrainer, CounterfactualTrainer, CombinedTrainer
from training.decoder_only.utils import setup_model_and_tokenizer, evaluate_model
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Decoder-Only Model")
    
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--group_name', type=str, default="debug")
    parser.add_argument('--dataset', type=str, choices=["cqa", "esnli", "strategyqa"])
    parser.add_argument('--model_type', type=str, required=True, choices=['task_prefix', 'counterfactual_prefix', 'both'])
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B", help="Model name or path")
    parser.add_argument("--train_data_path", default="../llm_outputs/consolidated/cqa-llama-train", help="Training data path")
    parser.add_argument("--test_data_path", default="../llm_outputs/consolidated/cqa-llama-test", help="Test data path")
    parser.add_argument("--target_rationale", default="few_shot_positive_rationale", help="Rationale column to use")
    parser.add_argument("--output_dir", default="../student_models/qwen-cf-cqa_v3", help="Output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluation steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="Multitask loss weight (predict vs explain)")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    
    # Usage
    wandb.init(
        project="model_distillation", 
        group=args.group_name,
        config=vars(args),
    )
    # Explicitly log the model_type to avoid confusion with model name
    wandb.config.update({"training_model_type": args.model_type})
    
    # Set seed
    seed = set_seed(args)
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    # Setup model and tokenizer
    print(f"Loading model: {args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(args.model_name)
    model = model.to(device)
    
    # Setup data
    print("Loading datasets...")
    if args.dataset == 'cqa':
        loader = CQADatasetLoader(sample=args.subsample)
    elif args.dataset == 'esnli':
        loader = ESNLIDatasetLoader(sample=args.subsample)
    elif args.dataset == 'strategyqa':
        loader = StrategyQADatasetLoader(sample=args.subsample)
    else:
        raise ValueError("Value for test dataset is not a valid name.")

    
    tokenized_train_datasets = loader.prepare_for_training(
        path=args.train_data_path,
        model_type=args.model_type,
        max_length=args.max_length,
        target_rationale=args.target_rationale,
        # turn off for decoder-only models since inputs 
        # and labels are concatenated in DataCollator
        tokenizer=tokenizer,
        truncation=False,
        padding=False,
    )
    
    tokenized_test_datasets = loader.prepare_for_training(
        path=args.test_data_path,
        model_type=args.model_type,
        max_length=args.max_length,
        target_rationale=args.target_rationale,
        tokenizer=tokenizer,
        # turn off for decoder-only models since inputs 
        # and labels are concatenated in DataCollator
        truncation=False,
        padding=False,
    )
    
    print(f"Train samples: {len(tokenized_train_datasets)}")
    print(f"Test samples: {len(tokenized_test_datasets)}")
    
    # Setup data collator
    if args.model_type == 'task_prefix':
        collator = MultitaskDataCollator(tokenizer=tokenizer)
    elif args.model_type == 'counterfactual_prefix':
        collator = CounterfactualDataCollator(tokenizer=tokenizer)
    elif args.model_type == 'both':
        collator = CombinedDataCollator(tokenizer=tokenizer)
    
    if args.eval_only:
        print("Evaluating model...")
        bootstrap_stats = evaluate_model(tokenized_test_datasets, model, tokenizer, collator, args.eval_batch_size)
        print(f"Bootstrapped stats: {bootstrap_stats}")
        return
    
    # Setup training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        prediction_loss_only=True
    )
    
    # Initialize trainer
    if args.model_type == 'task_prefix':
        trainer = MultitaskTrainer
    elif args.model_type == 'counterfactual_prefix':
        trainer = CounterfactualTrainer
    elif args.model_type == 'both':
        trainer = CombinedTrainer
    else: 
        raise ValueError("Model type must be 'task_prefix', 'counterfactual_prefix', or 'both'.")   
    
    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_test_datasets,
        tokenizer=tokenizer,
        data_collator=collator,
        alpha=args.alpha,
    )
    
    print("Starting training...")
    trainer.train()
    
    task="multitask_predict" if args.model_type in ["task_prefix", "both"] else "correct_answer"
    print(f"Evaluating final model on task (name of column after data preparation): {task}...")
    
    print("Training completed. Evaluating final model...")
    # We need more tokens for counterfactual prefix task due to longer outputs
    max_new_tokens = 128 if args.model_type == "counterfactual_prefix" else 16
    bootstrapped_stats = evaluate_model(
        data=tokenized_test_datasets, 
        model=trainer.model, 
        tokenizer=tokenizer, 
        collator=collator,
        max_new_tokens=max_new_tokens,
        batch_size=args.eval_batch_size,
        task=task,
    )
    print(f"Bootstrapped stats: {bootstrapped_stats}")
    
    # Log accuracy to wandb
    # trainer.log({"final_test_accuracy": accuracy})
    for bootstrap_key, bootstrap_value in bootstrapped_stats.items():
        wandb.log({f'test_{bootstrap_key}': bootstrap_value})
    
    # Save final model
    trainer.save_model()
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()