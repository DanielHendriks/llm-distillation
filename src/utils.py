import numpy as np
import wandb
import pandas as pd
from contextlib import contextmanager
from datetime import timedelta
import logging
import time, torch
from datasets import load_from_disk, concatenate_datasets
import os

def read_dataset(path, split=None):
    shard_dirs = [d for d in os.listdir(path) if d.startswith('shard_') and os.path.isdir(os.path.join(path, d))]
    if shard_dirs:
        num_shards = len(shard_dirs)
        return concatenate_datasets([
            load_from_disk(f"{path}/shard_{shard_idx}")
            for shard_idx in range(num_shards)])
    else:
        dataset = load_from_disk(path)
        return dataset[split] if split else dataset

@contextmanager
def timer(description, log_level=logging.INFO):
    """Context manager for timing code blocks with logging"""
    start_time = time.time()
    logging.log(log_level, f"⏳ Starting: {description}")
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        formatted_time = str(timedelta(seconds=round(elapsed_time, 2)))
        logging.log(log_level, f"✅ Completed: {description} | Duration: {formatted_time} ({elapsed_time:.2f}s)")


def set_seed(args: int):
    if not hasattr(args, 'seed') or args.seed is None:
        seed = np.random.randint(0, 100000)
    else: seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(args, 'n_gpu') and args.n_gpu > 0: torch.cuda.manual_seed_all(seed)
    return seed
        

### logging functions
def setup_wandb_logging(args, id=None, name=None):
    resume = None if id is None else "must"
    
    wandb.init(
        project="model-distillation",
        id=id,
        name=name,
        resume=resume,
        config={
        "subsample": args.subsample,
        "alpha": args.alpha,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "batch_size": args.batch_size,
        "optimizer_name": args.optimizer_name,
        "lr": args.lr,
        "run": args.run,
        "from_pretrained": args.from_pretrained,
        "max_input_length": args.max_input_length,
        "grad_steps": args.grad_steps,
        "local_rank": args.local_rank,
        "gen_max_len": args.gen_max_len,
        "model_type": args.model_type,
        "training_mode": str(args.model_type)+"_training", # new
        "output_rationale": args.output_rationale,
        "no_log": args.no_log,
        "max_length_tokenizer": args.max_length_tokenizer,
        "train_data_path": args.train_data_path,
        "test_data_path": args.test_data_path,
        }
    )

def log_test_sample(results_test):
    test_loss = results_test.metrics['test_loss']
    test_acc = results_test.metrics['test_accuracy']
    wandb.log({'test loss': test_loss, 'test accuracy': test_acc,})

def log_test_metrics(tokenized_test_dataset, tokenizer, results_test, model_type):
    
    # Handle different prediction structures based on model type
    if model_type == 'standard':
        # Standard model has single prediction output
        if isinstance(results_test.predictions, tuple):
            predictions = results_test.predictions[0]
        else:
            predictions = results_test.predictions
            
        answers_preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        answers_preds = tokenizer.batch_decode(answers_preds, skip_special_tokens=True)
        
        # No separate rationales for standard model
        rationales_preds = ["N/A"] * len(answers_preds)
        
        # Handle labels similarly
        if isinstance(results_test.label_ids, tuple):
            labels = results_test.label_ids[0]
        else:
            labels = results_test.label_ids
            
        answers_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        answers_labels = tokenizer.batch_decode(answers_labels, skip_special_tokens=True)
        rationales_labels = ["N/A"] * len(answers_labels)
    else:
        # Multitask models have separate answer and rationale predictions
        answers_preds = np.where(results_test.predictions[0] != -100, results_test.predictions[0], tokenizer.pad_token_id)
        answers_preds = tokenizer.batch_decode(answers_preds, skip_special_tokens=True)
        
        rationales_preds = np.where(results_test.predictions[1] != -100, results_test.predictions[1], tokenizer.pad_token_id)
        rationales_preds = tokenizer.batch_decode(rationales_preds, skip_special_tokens=True)

        answers_labels = np.where(results_test.label_ids[0] != -100, results_test.label_ids[0], tokenizer.pad_token_id)
        answers_labels = tokenizer.batch_decode(answers_labels, skip_special_tokens=True)
        
        rationales_labels = np.where(results_test.label_ids[1] != -100, results_test.label_ids[1], tokenizer.pad_token_id)
        rationales_labels = tokenizer.batch_decode(rationales_labels, skip_special_tokens=True)
        
    #answers_preds = tokenizer.batch_decode(results_test.predictions[0], skip_special_tokens=True)
    #rationales_preds = tokenizer.batch_decode(results_test.predictions[1], skip_special_tokens=True)

    #answers_labels = tokenizer.batch_decode(results_test.label_ids[0], skip_special_tokens=True)
    #rationales_labels = tokenizer.batch_decode(results_test.label_ids[1], skip_special_tokens=True)

    if model_type == 'counterfactual_prefix':
        test_input_predict = tokenizer.batch_decode(
            tokenized_test_dataset['correct_answer_input_encoded_input_ids'], 
            skip_special_tokens=True
            )
        
        test_input_explain = tokenizer.batch_decode(
            tokenized_test_dataset['false_answer1_input_encoded_input_ids'], 
            skip_special_tokens=True
            )
        
    elif model_type == 'standard':
        # Standard model uses basic input_ids column
        test_input_predict = tokenizer.batch_decode(
            tokenized_test_dataset['input_ids'], 
            skip_special_tokens=True
            )
        test_input_explain = ["N/A"] * len(test_input_predict)
        
    elif model_type == 'task_prefix' or model_type == 'both':
        test_input_predict = tokenizer.batch_decode(
            tokenized_test_dataset['multitask_predict_input_encoded_input_ids'], 
            skip_special_tokens=True
            )

        test_input_explain = tokenizer.batch_decode(
            tokenized_test_dataset['multitask_explain_input_encoded_input_ids'], 
            skip_special_tokens=True
            )

    test_samples = pd.DataFrame(
            {
            'input_predict': test_input_predict,
            'input_(counterfactual)explain': test_input_explain,
            'answers_preds': answers_preds, 
            'answers_labels': answers_labels, 
            'rationales_preds': rationales_preds, 
            'rationales_labels': rationales_labels, 
            }
    )

    test_samples_table = wandb.Table(dataframe=test_samples)
    wandb.log({"test_samples_table_revised": test_samples_table})
    