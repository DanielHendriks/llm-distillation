from src.utils import read_dataset
from datasets import load_from_disk
from typing import Any
import textgrad as tg
import os
from datasets import Dataset

tg.set_backward_engine(tg.get_engine("gpt-4o-mini"))

class TextGrader:
    def __call__(self, initiual_solution) -> str:
        return self.apply_textgrad(initiual_solution)

    def apply_textgrad(self, initial_solution):

        solution = tg.Variable(initial_solution,
                            requires_grad=True,
                            role_description="solution to the multiple-choice question")

        loss_system_prompt = tg.Variable("""You will evaluate an explanation to a multiple-choice question. 
        Do not attempt to answer it yourself, do not provide an explanation, only identify errors. Be super concise. 
        Only return the explanation, not the question, choices, or answer""",
                                        requires_grad=False,
                                        role_description="system prompt")
                                    
        loss_fn = tg.TextLoss(loss_system_prompt)
        optimizer = tg.TGD([solution])

        loss = loss_fn(solution)

        loss.backward()
        optimizer.step()
        return solution.value
    
    def process_batch(self, batch):
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        formatted = [format_example(e) for e in examples]
        graded = []
        for f in formatted:
            try:
                graded.append(safe_grade(self, f))
            except Exception as e:
                print(f"[process_batch] Failed permanently on example: {e}")
                graded.append(None)
        batch['graded_explanation'] = [
            extract_explanation(g) if g is not None else None for g in graded
        ]
        return batch

def format_example(example):
        return f"""Question: {example['question']}
Choices: {', '.join(example['choices'])}
Answer: {example['answer']}
Explanation: {example['few_shot_positive_rationale']}
"""

def extract_explanation(text): return text.split('Explanation: ')[-1]

def save_dataset_in_shards(dataset: Dataset, path_name: str, num_shards=10, overwrite=False):
    os.makedirs(path_name, exist_ok=True)
    for i in range(num_shards):
        shard_path = os.path.join(path_name, f'shard_{i}')
        if os.path.exists(shard_path) and not overwrite: continue
        shard = dataset.shard(num_shards=num_shards, index=i, contiguous=True)
        shard.save_to_disk(shard_path)

def save_dataset_shard(dataset: Dataset, path_name: str, shard_idx: int, overwrite=False):
    shard_dir = os.path.join(path_name, f'shard_{shard_idx}')
    os.makedirs(os.path.dirname(shard_dir), exist_ok=True)
    if os.path.exists(shard_dir) and not overwrite: return
    dataset.save_to_disk(shard_dir)


import time, random, functools

def robust(max_retries=5, base_delay=1.0, jitter=True, exceptions=(Exception,), verbose=False):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt + 1 == max_retries:
                        if verbose:
                            print(f"[robust] Giving up after {max_retries} attempts: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    if jitter:
                        delay *= random.uniform(0.5, 1.5)
                    if verbose:
                        print(f"[robust] Attempt {attempt+1} failed ({e}); retrying in {delay:.2f}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@robust(max_retries=5, base_delay=1.5, verbose=True)
def safe_grade(grader, text):
    return grader(text)

def process_batch(batch):
    grader = TextGrader()
    examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
    formatted = [format_example(e) for e in examples]
    graded = []
    for f in formatted:
        try:
            graded.append(safe_grade(grader, f))
        except Exception as e:
            print(f"[process_batch] Failed permanently on example: {e}")
            graded.append(None)
    batch['graded_explanation'] = [
        extract_explanation(g) if g is not None else None for g in graded
    ]
    return batch

def main(
    input_path = 'llm_outputs/rationales/cqa-llama-train',
    output_path = 'llm_outputs/rationales/cqa-textgraded-train',
    shard_idx = 0, 
    batch_size = 8,
    ):
    
    grader = TextGrader()
    ds = load_from_disk(f"{input_path}/shard_{shard_idx}")
    ds = ds.map(grader.process_batch, batched=True, batch_size=batch_size)
    save_dataset_shard(ds, output_path, shard_idx=shard_idx)