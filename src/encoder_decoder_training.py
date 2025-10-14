import argparse,  os, wandb, random, torch, logging
import numpy as np

from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from training.custom_metrics import compute_metrics_text

from dataloaders.cqa import CQADatasetLoader
from dataloaders.esnli import ESNLIDatasetLoader
from dataloaders.strategyqa import StrategyQADatasetLoader

from utils import setup_wandb_logging, log_test_sample, log_test_metrics, set_seed
from training.encoder_decoder.custom_datacollators import (
    MultitaskDataCollator, CounterfactualDataCollator, CounterfactualAndMultitaskDataCollator)
from training.encoder_decoder.custom_trainers import (
    MultitaskTrainer, CounterfactualTrainer, CounterfactualAndMultitaskTrainer)

os.environ['WANDB_WATCH'] = 'false'
os.environ["WANDB_LOG_MODEL"]= "false"
os.environ["WANDB_PROJECT"] = "model-distillation" 
hf_token = "<your_huggingface_token" 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


### train and evaluate
def train_and_evaluate(args, tokenizer, tokenized_datasets, compute_metrics, seed_dataset):
    seed = set_seed(args.seed)

    print("Model: ", args.from_pretrained)
    if "t5gemma" in args.from_pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.from_pretrained, token=hf_token, attn_implementation="eager")
    elif "t5" in args.from_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
    else: raise ValueError("Model not recognized")

    if not args.no_training:
        output_dir = f"../student_models/checkpoints/t5{args.from_pretrained.split('/')[-1].split('-')[-1]}-{args.model_type.replace('_', '-')}-{seed}-{args.target_rationale.replace('_', '-')}"  # for model ckpts
        logging_dir = f"../student_models/logs/{args.from_pretrained.split('/')[-1].split('-')[-1]}-{args.model_type.replace('_', '-')}-{seed}-{args.target_rationale.replace('_', '-')}"  # for training logs
    else:
        output_dir = f"../student_models/checkpoints/temporary"  # for model ckpts
        logging_dir = f"../student_models/logs/temporary"
        
    logging.info(f"Output directory: {output_dir}")    
    logging.info("Using bf16 precision instead of fp16 for T5 model compatibility")    
    
    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=True,
        eval_strategy = 'steps',
        eval_steps=args.eval_steps,
        save_strategy='no',
        save_steps=args.eval_steps,
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args.eval_steps,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=args.auto_batch_size, 
        predict_with_generate=True,
        seed=seed,
        local_rank=args.local_rank,
        generation_max_length=args.gen_max_len,
        prediction_loss_only=False,
        report_to=['wandb'],
        # run_name=run_name,
    )

    if args.model_type == 'task_prefix':
        data_collator = MultitaskDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    elif args.model_type == 'counterfactual_prefix':
        data_collator = CounterfactualDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    elif args.model_type == 'both':
        data_collator = CounterfactualAndMultitaskDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    else:
        raise ValueError

    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1, seed=seed_dataset)
    
    logging.info(f"After split - Train size: {len(tokenized_datasets['train'])}")
    logging.info(f"After split - Eval size: {len(tokenized_datasets['test'])}")

    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': tokenized_datasets["test"],
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }


    if args.model_type == 'task_prefix':
        trainer = MultitaskTrainer(**trainer_kwargs)
    elif args.model_type == 'counterfactual_prefix':
        trainer = CounterfactualTrainer(**trainer_kwargs)
    elif args.model_type == 'both':
        trainer_kwargs['beta'] = args.beta
        trainer_kwargs['gamma'] = args.gamma
        trainer = CounterfactualAndMultitaskTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError


    if not args.no_training:
        logging.info("Starting training...")
        logging.info(f"Trainer type: {type(trainer).__name__}")
        logging.info(f"Data collator type: {type(data_collator).__name__}")
        
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        
        trainer.train()
        logging.info("Training completed")
    else: 
        logging.debug("No training ...")
    
    return trainer


def test(trainer, tokenized_test_dataset):
    return trainer.predict(test_dataset=tokenized_test_dataset)


def run(args):
    # define tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, token=hf_token)

    if args.dataset == 'cqa':
        loader = CQADatasetLoader(sample=args.subsample)
    elif args.dataset == 'esnli':
        loader = ESNLIDatasetLoader(sample=args.subsample)
    elif args.dataset == 'strategyqa':
        loader = StrategyQADatasetLoader(sample=args.subsample)
    else:
        raise ValueError("Value for test dataset is not a valid name.")

    # load and prepare data for training with dataset_utils
    logging.info(f"Loading training data from: {args.train_data_path}")
    logging.info(f"Model type: {args.model_type}, Target rationale: {args.target_rationale}")
    
    tokenized_datasets = loader.prepare_for_training(
        path=args.train_data_path,
        model_type=args.model_type,
        tokenizer=tokenizer,
        max_length=args.max_length_tokenizer,
        target_rationale=args.target_rationale
    )
    
    logging.info(f"Loaded dataset - features: {tokenized_datasets.features}")
    logging.info(f"Dataset size: {len(tokenized_datasets)}")
    
    # Log a sample to understand the data structure
    sample = tokenized_datasets[0]
    logging.info(f"Sample keys: {sample.keys()}")
    logging.info(f"Input IDs shape: {len(sample.get('input_ids', []))}")
    logging.info(f"Labels shape: {len(sample.get('labels', []))}")
    
    # Decode and log sample input/output
    if 'input_ids' in sample:
        decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        logging.info(f"Sample input: {decoded_input[:200]}...")
    if 'labels' in sample:
        # Filter out -100 tokens for decoding
        labels_filtered = [token for token in sample['labels'] if token != -100]
        if labels_filtered:
            decoded_labels = tokenizer.decode(labels_filtered, skip_special_tokens=True)
            logging.info(f"Sample labels: {decoded_labels[:200]}...")

    # define metrics
    if args.model_type == 'counterfactual_prefix':
        # need to extract the final answer from the output to calculate the metrics
        do_extract = True 
    else:
        do_extract = False
        
    compute_metrics = compute_metrics_text(tokenizer, do_extract)

    # train and evaluate model
    trainer = train_and_evaluate(
        args, tokenizer, tokenized_datasets, compute_metrics, args.seed_dataset
        )

    # load and prepare data for testing with data_utils
    tokenized_test_dataset = loader.prepare_for_training(
        path=args.test_data_path,
        model_type=args.model_type,
        tokenizer=tokenizer,
        max_length=args.max_length_tokenizer,
        target_rationale=args.target_rationale
    )

    # test model
    results_test = test(trainer=trainer, tokenized_test_dataset=tokenized_test_dataset)

    # log test samples
    log_test_sample(results_test)

    # log test metrics
    log_test_metrics(tokenized_test_dataset, tokenizer, results_test, model_type=args.model_type)

    # save model from trainer
    if not args.no_training:
        trainer.save_model()
    elif args.no_training:
        logging.debug("No model saving, since no training was done. This is useful for testing only with the trainer class.")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=5_000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=300)
    parser.add_argument('--output_rationale', action='store_false', default=True, help='Set to avoid output rationale')
    parser.add_argument('--no_log', action='store_false')
    parser.add_argument('--max_length_tokenizer', type=int, default=300)
    parser.add_argument('--seed_dataset', type=int, default=0)
    parser.add_argument('--n_gpu', type=int, default=1)
    
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base') # google/t5-v1_1-large
    parser.add_argument('--model_type', type=str, required=True, choices=['standard', 'task_prefix', 'counterfactual_prefix', 'both'])
    parser.add_argument('--no_training', action='store_true', help="set if wanting to use test a model wihtout training it.")
    parser.add_argument('--train_data_path', type=str, default="../llm_outputs/consolidated/cqa-llama-train")
    parser.add_argument('--test_data_path', type=str, default="../llm_outputs/consolidated/cqa-llama-test")
    parser.add_argument('--target_rationale', type=str, default="few_shot_positive_rationale")
    parser.add_argument('--auto_batch_size', action='store_true')
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="cqa", choices=["cqa", "esnli", "strategyqa"])
    
    args = parser.parse_args()

    run(args)