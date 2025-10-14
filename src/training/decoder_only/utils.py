import numpy as np
import re, torch, scipy
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer for decoder-only training."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def generate_batch(model, tokenizer, inputs, max_new_tokens):
    """Generate responses for a batch of inputs."""
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        return model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


def get_extract_answer_fn(task):

    def extract_answer_mt(response, label):
        """Extract answer from model response."""
        return label.lower().strip() in response.lower().strip()

    def extract_answer_cf(response, label):
        """Extract the final answer after 'So the answer is'"""
        def _extract_final_answer(text):
            pattern = r"So the answer is (.+?)(?:\.|$)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            return matches[-1].strip() if matches else text.strip()
        
        # Extract answers from both texts
        response_answer = _extract_final_answer(response)
        label_answer = _extract_final_answer(label)
        
        # Compare case-insensitive
        return response_answer.lower().strip() == label_answer.lower().strip()

    if task == "multitask_predict":
        return extract_answer_mt 
    elif task == "correct_answer":
        return extract_answer_cf
    else: 
        raise ValueError(f"Unknown task: {task}")
    
    
def bootstrap(data, rng=42, confidence_level=0.95):
    bs = scipy.stats.bootstrap(
        (data, ), np.mean, n_resamples=10000, confidence_level=confidence_level,
        random_state=rng)
    return {
        'accuracy': np.mean(data).item(),
        'std_err': bs.standard_error,
        'low': bs.confidence_interval.low,
        'high': bs.confidence_interval.high
    }


def evaluate_model(data, model, tokenizer, collator, 
                   max_new_tokens=16, batch_size=8, 
                   task="multitask_predict"):
    """Evaluate model on dataset with generation-based metrics."""
    model.eval()
    results_all = []
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating on {len(data)} samples"):
        batch_indices = list(range(i, min(i + batch_size, len(data))))
        batch = data.select(batch_indices)
        collated_batch = collator(batch, concat_inputs_and_labels=False)
        
        # Generate responses
        outputs = generate_batch(model, tokenizer, collated_batch[task], max_new_tokens)
        new_tokens = outputs[:, collated_batch[task]["input_ids"].shape[-1]:]
        answer_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        # Get ground truth labels
        labels_text = tokenizer.batch_decode(batch[f"{task}_label_encoded_input_ids"], skip_special_tokens=True)
        
        # Evaluate batch
        extract_answer_fn = get_extract_answer_fn(task)
        results_batch = [extract_answer_fn(a, l) for a, l in zip(answer_text, labels_text)]
        results_all.extend(results_batch)
        
    bootstrap_stats = bootstrap(results_all)
    return bootstrap_stats