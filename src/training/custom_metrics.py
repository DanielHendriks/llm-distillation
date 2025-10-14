import numpy as np
import json 
import re


def compute_metrics_text(tokenizer, do_extract = False):
    # adapted from Hsieh el al. (2023)
    if do_extract:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # extract answer from label and predictions during counterfactual training 
            decoded_preds = [decoded_pred.split("So the answer is ")[-1].strip() for decoded_pred in decoded_preds]
            decoded_labels = [decoded_label.split("So the answer is ")[-1].strip() for decoded_label in decoded_labels] # "Answer: "

            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
                
            return {'accuracy': acc}
        
    else:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
                    
            return {'accuracy': acc}

    return compute_metrics


def compute_metrics_esnli(tokenizer, do_extract=False):
    """
    Compute evaluation metrics for ESNLI (e-SNLI) dataset.
    
    ESNLI uses NLI labels: entailment, neutral, contradiction
    Handles both direct label matching and extraction from "The answer is [label]" format.
    
    Args:
        tokenizer: HuggingFace tokenizer for decoding
        do_extract: If True, extract answer from "So the answer is" or "The answer is" format
    
    Returns:
        Function that computes accuracy for ESNLI predictions
    """
    
    # Valid ESNLI labels
    valid_labels = {'entailment', 'neutral', 'contradiction'}
    
    def extract_esnli_answer(text):
        """Extract NLI label from prediction text."""
        text = text.lower().strip()
        
        # Try "So the answer is" first (counterfactual format)
        if "so the answer is" in text:
            answer_part = text.split("so the answer is")[-1].strip()
        # Then try "The answer is" (standard format)
        elif "the answer is" in text:
            answer_part = text.split("the answer is")[-1].strip()
        else:
            answer_part = text
            
        # Clean up punctuation and whitespace
        answer_part = re.sub(r'[.,!?;:]', '', answer_part).strip()
        
        # Check for each valid label
        for label in valid_labels:
            if label in answer_part:
                return label
                
        # If no valid label found, return the cleaned text for fallback matching
        return answer_part
    
    if do_extract:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Extract ESNLI answers from predictions and labels
            extracted_preds = [extract_esnli_answer(pred) for pred in decoded_preds]
            extracted_labels = [extract_esnli_answer(label) for label in decoded_labels]
            
            acc = np.mean(np.array(extracted_preds) == np.array(extracted_labels))
            
            return {'accuracy': acc}
    else:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
            
            return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_strategyqa(tokenizer, do_extract=False):
    """
    Compute evaluation metrics for StrategyQA dataset.
    
    StrategyQA uses boolean labels: yes, no
    Handles both direct label matching and extraction from "The answer is [yes/no]" format.
    
    Args:
        tokenizer: HuggingFace tokenizer for decoding
        do_extract: If True, extract answer from "So the answer is" or "The answer is" format
    
    Returns:
        Function that computes accuracy for StrategyQA predictions
    """
    
    # Valid StrategyQA labels
    valid_labels = {'yes', 'no'}
    
    def extract_strategyqa_answer(text):
        """Extract yes/no answer from prediction text."""
        text = text.lower().strip()
        
        # Try "So the answer is" first (counterfactual format)
        if "so the answer is" in text:
            answer_part = text.split("so the answer is")[-1].strip()
        # Then try "The answer is" (standard format)
        elif "the answer is" in text:
            answer_part = text.split("the answer is")[-1].strip()
        else:
            answer_part = text
            
        # Clean up punctuation and whitespace
        answer_part = re.sub(r'[.,!?;:]', '', answer_part).strip()
        
        # Check for yes/no (prefer exact matches)
        if 'yes' in answer_part and 'no' not in answer_part:
            return 'yes'
        elif 'no' in answer_part and 'yes' not in answer_part:
            return 'no'
        elif answer_part.startswith('yes'):
            return 'yes'
        elif answer_part.startswith('no'):
            return 'no'
                
        # If no clear answer, return the cleaned text for fallback matching
        return answer_part
    
    if do_extract:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Extract StrategyQA answers from predictions and labels
            extracted_preds = [extract_strategyqa_answer(pred) for pred in decoded_preds]
            extracted_labels = [extract_strategyqa_answer(label) for label in decoded_labels]
            
            acc = np.mean(np.array(extracted_preds) == np.array(extracted_labels))
            
            return {'accuracy': acc}
    else:
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))
            
            return {'accuracy': acc}
    
    return compute_metrics


def get_compute_metrics_function(dataset_name, tokenizer, do_extract=False):
    """
    Get the appropriate compute_metrics function based on dataset name.
    
    Args:
        dataset_name: Name of dataset ('cqa', 'esnli', 'strategyqa')
        tokenizer: HuggingFace tokenizer
        do_extract: Whether to use extraction-based evaluation
    
    Returns:
        Appropriate compute_metrics function for the dataset
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['cqa', 'commonsenseqa']:
        return compute_metrics_text(tokenizer, do_extract)
    elif dataset_name in ['esnli', 'e-snli']:
        return compute_metrics_esnli(tokenizer, do_extract)
    elif dataset_name in ['strategyqa', 'strategy_qa']:
        return compute_metrics_strategyqa(tokenizer, do_extract)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: cqa, esnli, strategyqa")


